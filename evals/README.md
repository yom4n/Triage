# Triage Evals

Synthetic incidents in `evals/cases/` exercise the real triage API through
`TestClient(app)` and compare the response to ground truth.

Validate only the case files:

```bash
python -m evals.runner --dry-run
```

Run without a live LLM by forcing deterministic embeddings and the
rule-based fallback path:

```bash
python -m evals.runner --offline
```

Run the full scored eval against live infrastructure:

```bash
python -m evals.runner
```

Optional:

```bash
python -m evals.runner --limit 4 --offline
```

Scored runs write `evals/scorecard.json` for machines and `evals/REPORT.md`
for humans.

