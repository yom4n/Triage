"""End-to-end detector drill against demo-app, Prometheus, and backend."""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402


async def _set_fault(client: httpx.AsyncClient, demo_url: str, name: str, enabled: bool) -> None:
    response = await client.post(f"{demo_url}/admin/fault", json={"name": name, "enabled": enabled})
    response.raise_for_status()


async def _load(client: httpx.AsyncClient, demo_url: str, stop_at: float) -> None:
    while time.monotonic() < stop_at:
        try:
            await client.get(f"{demo_url}/work")
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.05)


async def _wait_for_metric_ticket(client: httpx.AsyncClient, backend_url: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get(f"{backend_url}/api/v1/tickets", params={"limit": 50})
        response.raise_for_status()
        for ticket in response.json():
            if ticket.get("source") == "metric":
                return ticket
        await asyncio.sleep(3)
    raise AssertionError("timed out waiting for source=metric ticket")


async def _wait_for_recovery(client: httpx.AsyncClient, backend_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get(f"{backend_url}/api/v1/detector/status")
        response.raise_for_status()
        rules = response.json().get("rules", [])
        if rules and all(not rule.get("firing") for rule in rules):
            return
        await asyncio.sleep(3)
    raise AssertionError("timed out waiting for detector rules to clear")


async def run(args: argparse.Namespace) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        await _set_fault(client, args.demo_url, args.fault, False)
        stop_at = time.monotonic() + args.load_seconds
        loaders = [asyncio.create_task(_load(client, args.demo_url, stop_at)) for _ in range(args.concurrency)]
        await asyncio.sleep(5)
        await _set_fault(client, args.demo_url, args.fault, True)
        ticket = await _wait_for_metric_ticket(client, args.backend_url, args.timeout)
        await _set_fault(client, args.demo_url, args.fault, False)
        await asyncio.gather(*loaders)
        await _wait_for_recovery(client, args.backend_url, args.timeout)
        print(f"Detection drill passed: metric ticket {ticket['ticket_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--demo-url", default="http://localhost:8081")
    parser.add_argument("--fault", default="errors", choices=["errors", "latency", "dependency_down"])
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--load-seconds", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        print(f"Detection drill failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
