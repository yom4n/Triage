"""Background Prometheus detector that files metric incidents through triage."""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI

from app.config import get_settings
from app.database import get_session_maker
from app.schemas import MetricAlert, metric_alert_to_ticket_create

logger = logging.getLogger("triage_engine.detector")


@dataclass(frozen=True)
class DetectorRule:
    name: str
    expr: str
    threshold_setting: str
    comparison: str
    severity_hint: str
    description: str


@dataclass
class RuleState:
    last_value: float | None = None
    firing: bool = False
    last_ticket_id: str | None = None
    last_fired_at: float = 0.0
    last_error: str | None = None


class DetectorService:
    """Owns the async polling task and in-memory per-rule firing state."""

    def __init__(self, app: FastAPI):
        self.app = app
        self.settings = get_settings()
        self.rules = [
            DetectorRule(
                name="high_error_rate",
                expr='sum(rate(demo_app_errors_total[1m]))',
                threshold_setting="detector_error_rate_threshold",
                comparison=">",
                severity_hint="HIGH",
                description="Demo app error rate is above threshold.",
            ),
            DetectorRule(
                name="high_p95_latency",
                expr='histogram_quantile(0.95, sum(rate(demo_app_request_latency_seconds_bucket[5m])) by (le))',
                threshold_setting="detector_p95_latency_threshold_seconds",
                comparison=">",
                severity_hint="MEDIUM",
                description="Demo app p95 latency is above threshold.",
            ),
            DetectorRule(
                name="dependency_down",
                expr="demo_app_dependency_up",
                threshold_setting="detector_dependency_down_threshold",
                comparison="<=",
                severity_hint="CRITICAL",
                description="Demo app dependency health metric indicates an outage.",
            ),
        ]
        self._states = {rule.name: RuleState() for rule in self.rules}
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if not self.settings.detector_enabled:
            return
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="prometheus-detector")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.detector_enabled,
            "rules": [
                {
                    "name": rule.name,
                    "expr": rule.expr,
                    "threshold": getattr(self.settings, rule.threshold_setting),
                    "comparison": rule.comparison,
                    "severity_hint": rule.severity_hint,
                    "last_value": self._states[rule.name].last_value,
                    "firing": self._states[rule.name].firing,
                    "last_ticket_id": self._states[rule.name].last_ticket_id,
                    "last_error": self._states[rule.name].last_error,
                }
                for rule in self.rules
            ],
        }

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.poll_once()
            except Exception:
                logger.exception("Detector poll failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.settings.detector_poll_seconds)
            except asyncio.TimeoutError:
                pass

    async def poll_once(self) -> None:
        async with httpx.AsyncClient(base_url=self.settings.prometheus_base_url, timeout=10.0) as client:
            for rule in self.rules:
                await self._poll_rule(client, rule)

    async def _poll_rule(self, client: httpx.AsyncClient, rule: DetectorRule) -> None:
        state = self._states[rule.name]
        try:
            response = await client.get("/api/v1/query", params={"query": rule.expr})
            response.raise_for_status()
            value, labels = self._extract_value(response.json())
            state.last_value = value
            state.last_error = None
        except Exception as exc:
            state.last_error = str(exc)
            logger.warning("Detector rule %s query failed: %s", rule.name, exc)
            return

        threshold = float(getattr(self.settings, rule.threshold_setting))
        breached = value > threshold if rule.comparison == ">" else value <= threshold
        now = time.time()
        if not breached:
            state.firing = False
            return
        if state.firing:
            return
        if now - state.last_fired_at < self.settings.detector_rule_cooldown_seconds:
            return

        state.firing = True
        state.last_fired_at = now
        alert = MetricAlert(
            rule_name=rule.name,
            expr=rule.expr,
            value=value,
            threshold=threshold,
            severity_hint=rule.severity_hint,
            description=rule.description,
            labels=labels,
        )
        await self._file_ticket(alert, state)

    async def _file_ticket(self, alert: MetricAlert, state: RuleState) -> None:
        from app.main import run_triage

        session_maker = get_session_maker()
        async with session_maker() as db:
            response = await run_triage(
                metric_alert_to_ticket_create(alert),
                graph=self.app.state.triage_graph,
                db=db,
                source="metric",
            )
            state.last_ticket_id = str(response.ticket_id)

    @staticmethod
    def _extract_value(body: dict[str, Any]) -> tuple[float, dict[str, str]]:
        result = body.get("data", {}).get("result", [])
        if not result:
            return 0.0, {}
        sample = result[0]
        value = sample.get("value", [None, 0])[1]
        labels = {str(k): str(v) for k, v in sample.get("metric", {}).items()}
        return float(value), labels
