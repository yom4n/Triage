import asyncio
import random
import time

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

app = FastAPI(title="Detector Demo App")

REQUESTS = Counter("demo_app_requests_total", "Total demo app requests", ["path"])
ERRORS = Counter("demo_app_errors_total", "Total demo app errors", ["path"])
LATENCY = Histogram(
    "demo_app_request_latency_seconds",
    "Demo app request latency",
    ["path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
DEPENDENCY_UP = Gauge("demo_app_dependency_up", "Whether the synthetic dependency is up")

FAULTS = {"latency": False, "errors": False, "dependency_down": False}
DEPENDENCY_UP.set(1)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/work")
async def work() -> dict[str, float]:
    REQUESTS.labels(path="/work").inc()
    started = time.perf_counter()
    try:
        if FAULTS["dependency_down"]:
            DEPENDENCY_UP.set(0)
        else:
            DEPENDENCY_UP.set(1)
        if FAULTS["latency"]:
            await asyncio.sleep(1.8 + random.random() * 0.4)
        else:
            await asyncio.sleep(0.02 + random.random() * 0.08)
        if FAULTS["errors"] and random.random() < 0.6:
            ERRORS.labels(path="/work").inc()
            raise HTTPException(status_code=500, detail="synthetic fault")
        return {"duration_s": time.perf_counter() - started}
    finally:
        LATENCY.labels(path="/work").observe(time.perf_counter() - started)


@app.post("/admin/fault")
async def set_fault(payload: dict[str, bool | str]) -> dict[str, dict[str, bool]]:
    name = str(payload.get("name", ""))
    enabled = bool(payload.get("enabled", False))
    if name not in FAULTS:
        raise HTTPException(status_code=400, detail=f"unknown fault: {name}")
    FAULTS[name] = enabled
    DEPENDENCY_UP.set(0 if FAULTS["dependency_down"] else 1)
    return {"faults": FAULTS}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
