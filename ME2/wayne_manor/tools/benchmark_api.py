from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.parse
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure local WayneManor API read latency")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8765/api/v1",
        help="WayneManor API root (default: %(default)s)",
    )
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=2.0)
    return parser.parse_args()


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    index = max(0, min(len(values) - 1, int(len(values) * percentile + 0.999999) - 1))
    return values[index]


def main() -> None:
    args = parse_args()
    parsed = urllib.parse.urlparse(args.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SystemExit("--base-url must be an HTTP or HTTPS URL")
    if args.requests <= 0 or args.timeout <= 0:
        raise SystemExit("--requests and --timeout must be positive")

    url = f"{args.base_url.rstrip('/')}/state"
    durations: list[float] = []
    for _ in range(args.requests):
        started = time.perf_counter()
        with urllib.request.urlopen(url, timeout=args.timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"Wayne Manor returned HTTP {response.status}")
            response.read()
        durations.append((time.perf_counter() - started) * 1_000)

    durations.sort()
    result = {
        "url": url,
        "requests": len(durations),
        "median_ms": round(statistics.median(durations), 3),
        "p95_ms": round(percentile_nearest_rank(durations, 0.95), 3),
        "maximum_ms": round(durations[-1], 3),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
