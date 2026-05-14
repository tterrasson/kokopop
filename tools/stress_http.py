#!/usr/bin/env uv run python
"""
Stress test for kokopop HTTP server (adaptative mode).

Usage:
    uv run tests/stress_http.py [--port 8080] [--host 127.0.0.1] \
        [--concurrency 10] [--total 50] [--voice af_heart]
"""

import argparse
import time
import urllib.request
import urllib.error
import json
import threading
import statistics

SAMPLE_TEXTS = [
    "Hello, this is a simple test sentence to stress the server.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "Artificial intelligence is transforming the way we interact with technology.",
    "In the middle of difficulty lies opportunity, so never give up.",
    "A journey of a thousand miles begins with a single step forward.",
    "She sells seashells by the seashore, every single day without rest.",
    "The weather today is absolutely beautiful with clear blue skies.",
    "Programming is both an art and a science combined into one.",
    "To be or not to be, that is the question Shakespeare asked.",
    "Music has a wonderful effect on the human mind and soul.",
]

# Track results
results = []
results_lock = threading.Lock()


def send_request(url: str, text: str, voice: str, idx: int):
    """Send a single TTS request and record timing."""
    payload = json.dumps({
        "text": text,
        "voice": voice,
        "mode": "adaptative",
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read()
            elapsed = time.perf_counter() - start
            status = resp.status

        with results_lock:
            results.append({
                "idx": idx,
                "status": status,
                "size_bytes": len(body),
                "elapsed_ms": elapsed * 1000,
                "text_len": len(text),
            })

        if status == 200 and len(body) > 100:
            print(f"  [OK] #{idx}  {status}  {elapsed*1000:8.1f}ms  {len(body):>8} bytes  ({len(text)} chars)")
        else:
            print(f"  [!!] #{idx}  {status}  {elapsed*1000:8.1f}ms  {len(body):>8} bytes")

    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - start
        body = e.read()
        with results_lock:
            results.append({
                "idx": idx,
                "status": e.code,
                "size_bytes": len(body),
                "elapsed_ms": elapsed * 1000,
                "text_len": len(text),
                "error": body.decode("utf-8", errors="replace")[:200],
            })
        print(f"  [ERR] #{idx}  {e.code}  {elapsed*1000:8.1f}ms  {body.decode('utf-8','replace')[:120]}")

    except Exception as e:
        elapsed = time.perf_counter() - start
        with results_lock:
            results.append({
                "idx": idx,
                "status": 0,
                "size_bytes": 0,
                "elapsed_ms": elapsed * 1000,
                "text_len": len(text),
                "error": str(e),
            })
        print(f"  [FAIL] #{idx}  {elapsed*1000:8.1f}ms  {e}")


def health_check(url: str):
    """Quick health check before the stress test."""
    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            print(f"Health: {data}")
            return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def print_report(results: list, wall_ms: float):
    """Print a summary report."""
    ok = [r for r in results if r["status"] == 200]
    err = [r for r in results if r["status"] != 200]

    if not ok:
        print("\nNo successful requests.")
        return

    latencies = [r["elapsed_ms"] for r in ok]
    sizes = [r["size_bytes"] for r in ok]

    print(f"\n{'='*60}")
    print("  STRESS TEST REPORT")
    print(f"{'='*60}")
    print(f"  Total requests      : {len(results)}")
    print(f"  OK (200)            : {len(ok)}")
    print(f"  Errors              : {len(err)}")
    print(f"  Wall time           : {wall_ms/1000:.2f}s")
    print(f"  Req/s (throughput)  : {len(ok) / (wall_ms/1000):.2f}")
    print("")
    print("  Latency (ms)")
    print(f"    min               : {min(latencies):.1f}")
    print(f"    p50               : {statistics.median(latencies):.1f}")
    print(f"    p95               : {sorted(latencies)[int(len(latencies)*0.95)]:.1f}")
    print(f"    p99               : {sorted(latencies)[int(len(latencies)*0.99)]:.1f}")
    print(f"    max               : {max(latencies):.1f}")
    print(f"    mean              : {statistics.mean(latencies):.1f}")
    print("")
    print("  Response size (bytes)")
    print(f"    avg               : {statistics.mean(sizes):.0f}")
    print(f"    min / max         : {min(sizes)} / {max(sizes)}")
    if err:
        print("\n  Error details:")
        for r in err:
            reason = r.get("error", f"HTTP {r['status']}")
            print(f"    #{r['idx']}: {reason[:120]}")


def main():
    parser = argparse.ArgumentParser(description="Stress test kokopop HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent requests")
    parser.add_argument("--total", type=int, default=50,
                        help="Total requests to send")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each sample text N times to artificially lengthen it")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially (no concurrency)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    tts_url = f"{base_url}/tts"

    # Warm-up
    print(f"Health check: {base_url}/health")
    if not health_check(base_url):
        print("Server may not be running. Starting anyway...")
    print()

    # Build request list
    requests_to_send = []
    for i in range(args.total):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] * args.repeat
        # Vary speed a bit
        requests_to_send.append({
            "idx": i,
            "text": text,
            "voice": args.voice,
        })

    # Send requests
    if args.sequential:
        print(f"Sequential mode: {args.total} requests")
        print(f"{'-'*60}")
        wall_start = time.perf_counter()
        for req in requests_to_send:
            send_request(tts_url, req["text"], req["voice"], req["idx"])
        wall_ms = (time.perf_counter() - wall_start) * 1000
        print_report(results, wall_ms)
        return

    # Concurrent mode with threading semaphore
    print(f"Concurrent mode: {args.total} requests, {args.concurrency} workers")
    print(f"{'-'*60}")
    semaphore = threading.Semaphore(args.concurrency)
    threads = []
    wall_start = time.perf_counter()

    def worker(req):
        try:
            semaphore.acquire()
            send_request(tts_url, req["text"], req["voice"], req["idx"])
        finally:
            semaphore.release()

    for req in requests_to_send:
        t = threading.Thread(target=worker, args=(req,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    wall_ms = (time.perf_counter() - wall_start) * 1000
    print_report(results, wall_ms)


if __name__ == "__main__":
    main()
