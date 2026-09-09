import argparse
import asyncio
import time

import httpx


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8081")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    stop_at = time.monotonic() + args.duration

    async def worker() -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < stop_at:
                try:
                    await client.get(f"{args.base_url}/work")
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.05)

    await asyncio.gather(*(worker() for _ in range(args.concurrency)))


if __name__ == "__main__":
    asyncio.run(main())
