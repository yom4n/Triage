"""
Run synthetic triage evaluations against the real FastAPI app.

Dry-run validation does not import app.main or start infrastructure. Scored
runs use TestClient(app) so the same route and lifespan wiring used by the
service is exercised.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from app.schemas import EnvironmentEnum, SeverityEnum, TicketCreate

ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = ROOT / "evals"
CASES_DIR = EVALS_DIR / "cases"
SCORECARD_PATH = EVALS_DIR / "scorecard.json"
REPORT_PATH = EVALS_DIR / "REPORT.md"


@dataclass(frozen=True)
class EvalCase:
    id: str
    title: str
    stack_trace: str
    environment: str
    description: str
    root_cause_keywords: list[str]
    severity: str
    should_escalate: bool


def _load_case_file(path: Path) -> EvalCase:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    required = {"id", "title", "stack_trace", "environment", "description", "expected"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{path.name}: missing required keys: {', '.join(sorted(missing))}")

    expected = raw["expected"]
    if not isinstance(expected, dict):
        raise ValueError(f"{path.name}: expected must be an object")

    expected_required = {"root_cause_keywords", "severity", "should_escalate"}
    expected_missing = expected_required - set(expected)
    if expected_missing:
        raise ValueError(f"{path.name}: missing expected keys: {', '.join(sorted(expected_missing))}")

    keywords = expected["root_cause_keywords"]
    if not isinstance(keywords, list) or not keywords or not all(isinstance(item, str) and item.strip() for item in keywords):
        raise ValueError(f"{path.name}: expected.root_cause_keywords must be a non-empty string list")

    severity = expected["severity"]
    if severity not in {item.value for item in SeverityEnum}:
        raise ValueError(f"{path.name}: expected.severity must be one of {[item.value for item in SeverityEnum]}")

    environment = raw["environment"]
    if environment not in {item.value for item in EnvironmentEnum}:
        raise ValueError(f"{path.name}: environment must be one of {[item.value for item in EnvironmentEnum]}")

    stack_trace = raw["stack_trace"]
    if not isinstance(stack_trace, str) or len(stack_trace.strip()) < 30:
        raise ValueError(f"{path.name}: stack_trace must contain at least 30 non-whitespace characters")

    try:
        TicketCreate(
            title=raw["title"],
            stack_trace=stack_trace,
            environment=environment,
            description=raw.get("description"),
        )
    except ValidationError as exc:
        raise ValueError(f"{path.name}: invalid TicketCreate payload: {exc}") from exc

    should_escalate = expected["should_escalate"]
    if not isinstance(should_escalate, bool):
        raise ValueError(f"{path.name}: expected.should_escalate must be boolean")

    return EvalCase(
        id=raw["id"],
        title=raw["title"],
        stack_trace=stack_trace,
        environment=environment,
        description=raw["description"],
        root_cause_keywords=[item.strip() for item in keywords],
        severity=severity,
        should_escalate=should_escalate,
    )


def load_cases(limit: int | None = None) -> list[EvalCase]:
    paths = sorted(CASES_DIR.glob("*.json"))
    if not paths:
        raise ValueError(f"no eval case files found in {CASES_DIR}")
    cases = [_load_case_file(path) for path in paths]
    ids = [case.id for case in cases]
    duplicate_ids = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicate_ids:
        raise ValueError(f"duplicate eval case ids: {', '.join(duplicate_ids)}")
    return cases[:limit] if limit is not None else cases


def _payload(case: EvalCase) -> dict[str, Any]:
    return {
        "title": case.title,
        "stack_trace": case.stack_trace,
        "environment": case.environment,
        "description": case.description,
    }


def _joined_diagnosis(body: dict[str, Any]) -> str:
    parts: list[str] = [
        str(body.get("extracted_error") or ""),
        str(body.get("summary") or ""),
    ]
    steps = body.get("resolution_steps") or []
    if isinstance(steps, list):
        parts.extend(str(step) for step in steps)
    return "\n".join(parts).lower()


def _score_case(case: EvalCase, body: dict[str, Any]) -> dict[str, Any]:
    diagnosis = _joined_diagnosis(body)
    matched_keywords = [keyword for keyword in case.root_cause_keywords if keyword.lower() in diagnosis]
    root_cause_hit = bool(matched_keywords)
    status = body.get("status")
    fix_attempted = bool(body.get("fix_attempted"))
    fix_status = str(body.get("fix_verification_status") or "NOT_ATTEMPTED")
    passed = fix_status == "PASSED"
    attempts = int(body.get("fix_verification_attempts") or 0)
    return {
        "id": case.id,
        "title": case.title,
        "http_status": 201,
        "root_cause_hit": root_cause_hit,
        "matched_keywords": matched_keywords,
        "expected_keywords": case.root_cause_keywords,
        "severity_expected": case.severity,
        "severity_actual": body.get("severity"),
        "severity_match": body.get("severity") == case.severity,
        "should_escalate_expected": case.should_escalate,
        "should_escalate_actual": status == "ESCALATED_TO_HUMAN",
        "escalation_correct": (status == "ESCALATED_TO_HUMAN") == case.should_escalate,
        "status": status,
        "fix_attempted": fix_attempted,
        "fix_verification_status": fix_status,
        "fix_verification_attempts": attempts,
        "fix_verified": passed,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    predicted_escalations = sum(1 for item in results if item["should_escalate_actual"])
    expected_escalations = sum(1 for item in results if item["should_escalate_expected"])
    true_escalations = sum(
        1 for item in results if item["should_escalate_actual"] and item["should_escalate_expected"]
    )
    precision = true_escalations / predicted_escalations if predicted_escalations else 0.0
    recall = true_escalations / expected_escalations if expected_escalations else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    attempted = [item for item in results if item["fix_attempted"]]
    passed = [item for item in attempted if item["fix_verification_status"] == "PASSED"]
    return {
        "case_count": total,
        "root_cause_hit_rate": sum(1 for item in results if item["root_cause_hit"]) / total if total else 0.0,
        "severity_accuracy": sum(1 for item in results if item["severity_match"]) / total if total else 0.0,
        "escalation": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_escalations,
            "predicted": predicted_escalations,
            "expected": expected_escalations,
        },
        "fix_verification_rate": len(passed) / len(attempted) if attempted else 0.0,
        "fix_attempted_count": len(attempted),
        "fix_verification_passed_count": len(passed),
        "mean_fix_verification_attempts_over_passed": statistics.fmean(
            item["fix_verification_attempts"] for item in passed
        )
        if passed
        else 0.0,
    }


def _write_outputs(scorecard: dict[str, Any]) -> None:
    SCORECARD_PATH.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Triage Eval Report",
        "",
        f"Generated: {scorecard['generated_at']}",
        f"Cases: {scorecard['aggregate']['case_count']}",
        "",
        "| Case | Root Cause | Severity | Escalation | Fix Verify |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in scorecard["cases"]:
        lines.append(
            "| {id} | {root} | {sev_actual}/{sev_expected} | {esc_actual}/{esc_expected} | {fix} |".format(
                id=item["id"],
                root="PASS" if item["root_cause_hit"] else "FAIL",
                sev_actual=item["severity_actual"],
                sev_expected=item["severity_expected"],
                esc_actual="YES" if item["should_escalate_actual"] else "NO",
                esc_expected="YES" if item["should_escalate_expected"] else "NO",
                fix=item["fix_verification_status"],
            )
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scored(cases: list[EvalCase], offline: bool) -> dict[str, Any]:
    if offline:
        os.environ["EMBEDDING_PROVIDER"] = "deterministic"

    from fastapi.testclient import TestClient

    import app.graph as graph_module
    from app.main import app

    async def _offline_llm_failure(**kwargs):
        raise RuntimeError("offline eval forced rule-based fallback")

    patcher = patch.object(graph_module, "call_structured", new=AsyncMock(side_effect=_offline_llm_failure))
    context = patcher if offline else _NullContext()
    results: list[dict[str, Any]] = []
    with context:
        with TestClient(app) as client:
            for case in cases:
                response = client.post("/api/v1/triage", json=_payload(case))
                if response.status_code != 201:
                    results.append(
                        {
                            "id": case.id,
                            "title": case.title,
                            "http_status": response.status_code,
                            "error": response.text,
                            "root_cause_hit": False,
                            "severity_match": False,
                            "escalation_correct": False,
                            "should_escalate_expected": case.should_escalate,
                            "should_escalate_actual": False,
                            "fix_attempted": False,
                            "fix_verification_status": "NOT_ATTEMPTED",
                            "fix_verification_attempts": 0,
                        }
                    )
                    continue
                results.append(_score_case(case, response.json()))

    scorecard = {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline" if offline else "full",
        "aggregate": _aggregate(results),
        "cases": results,
    }
    _write_outputs(scorecard)
    return scorecard


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic evals against the real triage API.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N sorted cases.")
    parser.add_argument("--dry-run", action="store_true", help="Validate case files without starting the app.")
    parser.add_argument("--offline", action="store_true", help="Use deterministic embeddings and force LLM fallback.")
    args = parser.parse_args()

    cases = load_cases(args.limit)
    if args.dry_run:
        print(f"validated {len(cases)} eval case(s)")
        return

    scorecard = run_scored(cases, args.offline)
    aggregate = scorecard["aggregate"]
    print(
        "eval complete: root_cause={:.1%} severity={:.1%} escalation_f1={:.1%} fix_verify={:.1%}".format(
            aggregate["root_cause_hit_rate"],
            aggregate["severity_accuracy"],
            aggregate["escalation"]["f1"],
            aggregate["fix_verification_rate"],
        )
    )


if __name__ == "__main__":
    main()

