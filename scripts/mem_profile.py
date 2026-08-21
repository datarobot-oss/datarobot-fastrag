#!/usr/bin/env python3
"""
Local memory profiler for a containerised custom model runner (fastrag or DRUM).

Runs one container under a fixed memory limit, samples the container's cgroup
memory while driving chat traffic at it, and reports:

  cold baseline   memory right after load_model, zero traffic  -> OOM/migration risk
  sweep           peak memory at each concurrency level        -> where the curve bends
  MB per RPS      total footprint divided by achieved rps      -> efficiency vs DRUM
  leak slope      memory drift over a long soak                -> per-request leaks
  retained        memory left over after traffic stops         -> what never comes back

Memory is read from the container's own cgroup (v2 preferred, v1 fallback), which
is the number the OOM killer acts on. Two series are recorded:

  anon      anonymous (heap) pages - the fairest cross-runtime metric
  current   everything charged to the cgroup, incl. page cache - the limit's view

Usage:
    # full run: build wheel + image, profile, report
    uv run scripts/mem_profile.py --model-dir tests/benchmark_model_rag

    # reuse an already-built image, quick pass
    uv run scripts/mem_profile.py --no-build --phase-seconds 8 --soak-requests 2000

    # DRUM side of the comparison (same image, sync model)
    uv run scripts/mem_profile.py --runner drum --no-build \
        --model-dir tests/benchmark_model_rag_sync

Prerequisites: Docker running, and the base image pulled:
    docker pull datarobotdev/buzok-genai-custom-model-local-dropin-env:latest
"""

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_BASE_IMAGE = "datarobotdev/buzok-genai-custom-model-local-dropin-env:latest"
DEFAULT_IMAGE = "fastrag-mem-test"
CONTAINER = "fastrag-memprofile"
MB = 1024.0 * 1024.0

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
NC = "\033[0m"


def info(msg):
    print(f"  {YELLOW}->{NC} {msg}", flush=True)


def ok(msg):
    print(f"  {GREEN}v{NC} {msg}", flush=True)


def bad(msg):
    print(f"  {RED}x{NC} {msg}", flush=True)


# ---------------------------------------------------------------------------
# In-container sampler
# ---------------------------------------------------------------------------

# Marker string lets us pkill the loop if the container is kept alive.
SAMPLER_SH = r"""
: memprofile-sampler
while :; do
  if [ -r /sys/fs/cgroup/memory.current ]; then
    cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    anon=$(grep -m1 '^anon ' /sys/fs/cgroup/memory.stat 2>/dev/null | cut -d' ' -f2)
    if [ -r /sys/fs/cgroup/memory.peak ]; then
      pk=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0)
    else
      pk=0
    fi
  else
    cur=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
    anon=$(grep -m1 '^total_rss ' /sys/fs/cgroup/memory/memory.stat 2>/dev/null | cut -d' ' -f2)
    pk=$(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null || echo 0)
  fi
  echo "${cur:-0} ${anon:-0} ${pk:-0}"
  sleep %(interval)s
done
"""

PROC_SH = r"""
for d in /proc/[0-9]*; do
  [ -r "$d/statm" ] || continue
  rss=$(cut -d' ' -f2 "$d/statm" 2>/dev/null) || continue
  [ -n "$rss" ] || continue
  thr=$(grep -m1 '^Threads:' "$d/status" 2>/dev/null | tr -s '\t ' ' ' | cut -d' ' -f2)
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null | cut -c1-70)
  echo "$(basename "$d")|${rss}|${thr:-?}|${cmd}"
done
"""


class Sampler:
    """Streams cgroup memory samples out of the container via one long-lived exec."""

    def __init__(self, container, interval=0.25):
        self.container = container
        self.interval = interval
        self.samples = []  # (host_time, current_bytes, anon_bytes, peak_bytes)
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        script = SAMPLER_SH % {"interval": self.interval}
        self._proc = subprocess.Popen(
            ["docker", "exec", self.container, "sh", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        # Make sure at least one sample lands before callers start slicing windows.
        deadline = time.time() + 10
        while not self.samples and time.time() < deadline:
            time.sleep(0.1)
        if not self.samples:
            raise RuntimeError(
                "cgroup sampler produced no samples; is /sys/fs/cgroup readable in the container?"
            )

    def _read(self):
        for line in self._proc.stdout:
            if self._stop.is_set():
                return
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                cur, anon, peak = (int(p) for p in parts)
            except ValueError:
                continue
            self.samples.append((time.time(), cur, anon, peak))

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        # The exec'd shell survives the client; kill it in case --keep is set.
        subprocess.run(
            ["docker", "exec", self.container, "pkill", "-f", "memprofile-sampler"],
            capture_output=True,
        )

    def window(self, t0, t1):
        return [s for s in self.samples if t0 <= s[0] <= t1]

    def peak_bytes(self):
        """Kernel-tracked high-water mark, immune to sampling gaps (0 if unsupported)."""
        return max((s[3] for s in self.samples), default=0)


def summarize_window(samples):
    if not samples:
        return None
    anon = [s[2] / MB for s in samples]
    cur = [s[1] / MB for s in samples]
    return {
        "n_samples": len(samples),
        "anon_mean_mb": statistics.fmean(anon),
        "anon_max_mb": max(anon),
        "anon_min_mb": min(anon),
        "current_mean_mb": statistics.fmean(cur),
        "current_max_mb": max(cur),
    }


# ---------------------------------------------------------------------------
# Docker plumbing
# ---------------------------------------------------------------------------


def sh(cmd, check=True, capture=True):
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def require_docker():
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        bad("Docker daemon is not reachable. Start Docker Desktop and retry.")
        sys.exit(1)


def image_exists(name):
    res = subprocess.run(["docker", "image", "inspect", name], capture_output=True)
    return res.returncode == 0


def build_wheel():
    info("Building wheel (uv build --wheel)...")
    sh(["uv", "build", "--wheel", "-q"])
    wheels = sorted((REPO_ROOT / "dist").glob("datarobot_fastrag-*.whl"))
    if not wheels:
        bad("No wheel found in dist/ after build.")
        sys.exit(1)
    ok(f"Wheel: {wheels[-1].name}")


def build_image(base_image, tag):
    if not image_exists(base_image):
        bad(f"Base image not present locally: {base_image}")
        print(f"      docker pull {base_image}")
        sys.exit(1)
    info(f"Building image {tag} from {base_image}...")
    sh(["docker", "build", "-f", "Dockerfile.local-test", "-t", tag, "--quiet", "."])
    ok(f"Image built: {tag}")


def runner_cmd(runner, target_type):
    if runner == "fastrag":
        return (
            ["fastrag"],
            ["server", "--code-dir", "/opt/model", "--address", "0.0.0.0:8080"],
        )
    # DRUM side of the same comparison. Untested here - verify before trusting numbers.
    return (
        ["drum"],
        [
            "server",
            "--code-dir",
            "/opt/model",
            "--address",
            "0.0.0.0:8080",
            "--target-type",
            target_type,
        ],
    )


def start_container(args, model_dir):
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    entrypoint, cmd = runner_cmd(args.runner, args.target_type)

    env = {
        "TARGET_TYPE": args.target_type,
        # DRUM requires TARGET_NAME in the env for textgeneration targets
        # (python_model_adapter.py); fastrag defaults it. Set on both runners so
        # the two containers see an identical environment.
        "TARGET_NAME": args.target_name,
        "MAX_WORKERS": str(args.max_workers),
        "PYTHONUNBUFFERED": "1",
    }
    if args.malloc_arena_max:
        env["MALLOC_ARENA_MAX"] = str(args.malloc_arena_max)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    run_cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER,
        "--memory", args.memory,
        "--memory-swap", args.memory,
        "-p", f"{args.port}:8080",
        "-v", f"{model_dir}:/opt/model",
        "--entrypoint", entrypoint[0],
    ]
    for k, v in env.items():
        run_cmd += ["-e", f"{k}={v}"]
    run_cmd += [args.image] + cmd

    info(f"Starting {args.runner} container (limit={args.memory}, MAX_WORKERS={args.max_workers})")
    print(f"      {DIM}{' '.join(run_cmd)}{NC}")
    sh(run_cmd)


# fastrag serves /ping/ and /health/; DRUM only serves /health/.
READY_PATHS = ("/ping/", "/health/")


def wait_ready(base_url, timeout):
    info(f"Waiting for readiness (up to {timeout}s)...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        for path in READY_PATHS:
            try:
                r = httpx.get(f"{base_url}{path}", timeout=2.0)
                if r.status_code == 200:
                    elapsed = time.time() - t0
                    ok(f"Ready in {elapsed:.1f}s (via {path})")
                    return elapsed
            except Exception:
                pass
        if subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
            capture_output=True, text=True,
        ).stdout.strip() != "true":
            bad("Container exited during startup. Logs:")
            print(logs(tail=40))
            sys.exit(1)
        time.sleep(1)
    bad("Server did not become ready in time. Logs:")
    print(logs(tail=40))
    sys.exit(1)


def logs(tail=50):
    res = subprocess.run(
        ["docker", "logs", "--tail", str(tail), CONTAINER], capture_output=True, text=True
    )
    return res.stdout + res.stderr


def proc_snapshot():
    res = subprocess.run(
        ["docker", "exec", CONTAINER, "sh", "-c", PROC_SH], capture_output=True, text=True
    )
    rows = []
    for line in res.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        pid, rss_pages, threads, cmd = parts
        try:
            rss_mb = int(rss_pages) * 4096 / MB
        except ValueError:
            continue
        rows.append({"pid": pid, "rss_mb": rss_mb, "threads": threads, "cmd": cmd.strip()})
    return sorted(rows, key=lambda r: -r["rss_mb"])


def cleanup(keep):
    if keep:
        print()
        info(f"Container left running (--keep): docker logs {CONTAINER}")
        return
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


# ---------------------------------------------------------------------------
# Load driver
# ---------------------------------------------------------------------------


async def drive(url, payload, concurrency, duration_s=None, requests=None, timeout=60.0):
    """Hold `concurrency` requests in flight until a deadline or a request count."""
    latencies = []
    statuses = {}
    errors = []
    state = {"remaining": requests}
    deadline = time.monotonic() + duration_s if duration_s else None
    limits = httpx.Limits(
        max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8
    )

    t0 = time.time()
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:

        async def worker():
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if state["remaining"] is not None:
                    if state["remaining"] <= 0:
                        return
                    state["remaining"] -= 1
                started = time.monotonic()
                try:
                    resp = await client.post(url, json=payload)
                    _ = resp.content
                    statuses[resp.status_code] = statuses.get(resp.status_code, 0) + 1
                    if resp.status_code == 200:
                        latencies.append(time.monotonic() - started)
                    else:
                        errors.append(f"HTTP {resp.status_code}: {resp.text[:200]}")
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")

        await asyncio.gather(*[worker() for _ in range(concurrency)])
    t1 = time.time()

    n_ok = len(latencies)
    total = n_ok + len(errors)
    elapsed = max(t1 - t0, 1e-9)
    qs = statistics.quantiles(latencies, n=100) if n_ok > 1 else [latencies[0]] * 99 if n_ok else []
    return {
        "concurrency": concurrency,
        "t0": t0,
        "t1": t1,
        "elapsed_s": elapsed,
        "requests": total,
        "ok": n_ok,
        "errors": len(errors),
        "rps": total / elapsed,
        "p50_ms": qs[49] * 1000 if qs else 0.0,
        "p95_ms": qs[94] * 1000 if qs else 0.0,
        "statuses": statuses,
        "error_sample": errors[0] if errors else None,
    }


def idle(seconds, label):
    info(f"{label}: idling {seconds}s")
    t0 = time.time()
    time.sleep(seconds)
    return {"t0": t0, "t1": time.time()}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def linreg_slope(xs, ys):
    """Least-squares slope of ys over xs; 0.0 when undetermined."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def analyse_soak(sampler, load, phase):
    samples = sampler.window(phase["t0"], phase["t1"])
    if len(samples) < 5:
        return None
    base_t = samples[0][0]
    xs = [s[0] - base_t for s in samples]
    ys = [s[2] / MB for s in samples]
    slope_mb_per_s = linreg_slope(xs, ys)
    rps = load["rps"] if load["rps"] > 0 else 1.0
    quarter = max(len(samples) // 4, 1)
    first_q = statistics.fmean(ys[:quarter])
    last_q = statistics.fmean(ys[-quarter:])
    return {
        "slope_mb_per_s": slope_mb_per_s,
        "slope_mb_per_1k_requests": slope_mb_per_s / rps * 1000.0,
        "first_quarter_mean_mb": first_q,
        "last_quarter_mean_mb": last_q,
        "quarter_delta_mb": last_q - first_q,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runner", choices=["fastrag", "drum"], default="fastrag")
    p.add_argument("--model-dir", default="tests/benchmark_model_rag",
                   help="Model code dir mounted at /opt/model (default: tests/benchmark_model_rag)")
    p.add_argument("--image", default=DEFAULT_IMAGE, help="Image to run")
    p.add_argument("--base-image", default=DEFAULT_BASE_IMAGE, help="Base image for the build")
    p.add_argument("--no-build", action="store_true", help="Skip wheel + image build")
    p.add_argument("--memory", default="2g", help="Container memory limit (default: 2g)")
    p.add_argument("--max-workers", type=int, default=1, help="MAX_WORKERS (default: 1)")
    p.add_argument("--malloc-arena-max", type=int, default=0,
                   help="Set MALLOC_ARENA_MAX (0 = leave unset, matching production)")
    p.add_argument("--target-type", default="textgeneration")
    p.add_argument("--target-name", default="target")
    p.add_argument("--port", type=int, default=8086)
    p.add_argument("--env", action="append", default=[], metavar="K=V",
                   help="Extra env var for the container (repeatable)")
    p.add_argument("--ready-timeout", type=int, default=180)
    p.add_argument("--sample-interval", type=float, default=0.25)
    p.add_argument("--cold-seconds", type=int, default=20,
                   help="Idle window right after readiness (cold baseline)")
    p.add_argument("--warmup-requests", type=int, default=200)
    p.add_argument("--baseline-seconds", type=int, default=15,
                   help="Idle window after warmup (warm baseline)")
    p.add_argument("--concurrency", default="1,2,4,8,16,32,64,128")
    p.add_argument("--phase-seconds", type=int, default=15, help="Duration per sweep level")
    p.add_argument("--settle", type=int, default=3, help="Idle gap between phases")
    p.add_argument("--soak-requests", type=int, default=10000, help="0 to skip the soak")
    p.add_argument("--soak-concurrency", type=int, default=16)
    p.add_argument("--cooldown-seconds", type=int, default=30,
                   help="Idle window after the soak (retained memory)")
    p.add_argument("--out", default=".mem-profile", help="Output directory for JSON + samples")
    p.add_argument("--keep", action="store_true", help="Leave the container running afterwards")
    return p.parse_args()


def main():
    args = parse_args()
    require_docker()

    model_dir = (REPO_ROOT / args.model_dir).resolve() if not Path(args.model_dir).is_absolute() \
        else Path(args.model_dir)
    if not (model_dir / "custom.py").exists():
        bad(f"No custom.py in {model_dir}")
        sys.exit(1)

    base_url = f"http://localhost:{args.port}"
    chat_url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "memprofile",
        "messages": [{"role": "user", "content": "hello"}],
    }
    levels = [int(c) for c in args.concurrency.split(",") if c.strip()]

    print()
    print(f"{args.runner} memory profile")
    print("=" * 72)
    print(f"  model-dir   {model_dir}")
    print(f"  image       {args.image}")
    print(f"  mem limit   {args.memory}   MAX_WORKERS={args.max_workers}"
          f"   MALLOC_ARENA_MAX={args.malloc_arena_max or 'unset'}")
    print()

    if not args.no_build:
        build_wheel()
        build_image(args.base_image, args.image)
    elif not image_exists(args.image):
        bad(f"Image {args.image} not found and --no-build was passed.")
        sys.exit(1)

    sampler = None
    report = {
        "runner": args.runner,
        "model_dir": str(model_dir),
        "memory_limit": args.memory,
        "max_workers": args.max_workers,
        "malloc_arena_max": args.malloc_arena_max or None,
        "phases": {},
        "sweep": [],
    }

    try:
        start_container(args, model_dir)
        report["startup_seconds"] = wait_ready(base_url, args.ready_timeout)

        sampler = Sampler(CONTAINER, args.sample_interval)
        sampler.start()
        ok(f"cgroup sampler running ({args.sample_interval}s interval)")
        print()

        # 1. cold baseline - after load_model, before any traffic
        cold = idle(args.cold_seconds, "cold baseline")
        report["phases"]["cold"] = summarize_window(sampler.window(cold["t0"], cold["t1"]))

        # 2. warmup, discarded
        if args.warmup_requests:
            info(f"warmup: {args.warmup_requests} requests @ c=8")
            w = asyncio.run(drive(chat_url, payload, 8, requests=args.warmup_requests))
            if w["errors"]:
                bad(f"warmup had {w['errors']} error(s): {w['error_sample']}")
                print(logs(tail=30))
                sys.exit(1)
            ok(f"warmup done: {w['rps']:.1f} rps, p95={w['p95_ms']:.0f}ms")

        # 3. warm baseline
        baseline = idle(args.baseline_seconds, "warm baseline")
        report["phases"]["baseline"] = summarize_window(
            sampler.window(baseline["t0"], baseline["t1"])
        )
        base_anon = report["phases"]["baseline"]["anon_mean_mb"]

        # 4. concurrency sweep
        print()
        info(f"concurrency sweep: {levels}  ({args.phase_seconds}s each)")
        for c in levels:
            load = asyncio.run(drive(chat_url, payload, c, duration_s=args.phase_seconds))
            mem = summarize_window(sampler.window(load["t0"], load["t1"]))
            row = {"load": load, "mem": mem}
            report["sweep"].append(row)
            delta = mem["anon_max_mb"] - base_anon if mem else 0.0
            print(f"    c={c:<4} {load['rps']:7.1f} rps  p95={load['p95_ms']:6.0f}ms  "
                  f"anon peak {mem['anon_max_mb']:7.1f} MB  (+{delta:6.1f})  "
                  f"err={load['errors']}")
            if load["errors"]:
                print(f"         {RED}error sample:{NC} {load['error_sample']}")
            time.sleep(args.settle)

        report["procs_after_sweep"] = proc_snapshot()

        # 5. soak - leak detection
        if args.soak_requests:
            print()
            info(f"soak: {args.soak_requests} requests @ c={args.soak_concurrency}")
            soak = asyncio.run(
                drive(chat_url, payload, args.soak_concurrency, requests=args.soak_requests)
            )
            report["phases"]["soak"] = summarize_window(sampler.window(soak["t0"], soak["t1"]))
            report["soak_load"] = soak
            report["leak"] = analyse_soak(sampler, soak, soak)
            ok(f"soak done: {soak['rps']:.1f} rps, {soak['errors']} errors, "
               f"{soak['elapsed_s']:.0f}s")

        # 6. cooldown - retained memory
        print()
        cool = idle(args.cooldown_seconds, "cooldown")
        report["phases"]["cooldown"] = summarize_window(sampler.window(cool["t0"], cool["t1"]))
        report["kernel_peak_mb"] = sampler.peak_bytes() / MB

    finally:
        if sampler:
            sampler.stop()
            outdir = REPO_ROOT / args.out
            outdir.mkdir(parents=True, exist_ok=True)
            # Timestamped so a later run never clobbers an earlier one's raw samples.
            stamp = time.strftime("%Y%m%d-%H%M%S")
            tsv = outdir / f"samples-{args.runner}-{stamp}.tsv"
            with tsv.open("w") as fh:
                fh.write("host_time\tcurrent_bytes\tanon_bytes\tpeak_bytes\n")
                for s in sampler.samples:
                    fh.write("\t".join(str(x) for x in s) + "\n")
            report["samples_file"] = str(tsv)
            report["run_stamp"] = stamp
            payload = json.dumps(report, indent=2)
            (outdir / f"report-{args.runner}-{stamp}.json").write_text(payload)
            (outdir / f"report-{args.runner}-latest.json").write_text(payload)
        cleanup(args.keep)

    print_report(report, args)


def print_report(report, args):
    ph = report["phases"]
    print()
    print("=" * 72)
    print(f"{report['runner']}  |  limit {report['memory_limit']}  |  "
          f"MAX_WORKERS={report['max_workers']}")
    print("=" * 72)

    print("\nBaselines (no traffic)")
    print("-" * 72)
    for key, label in (("cold", "cold (post load_model)"), ("baseline", "warm (post warmup)"),
                       ("cooldown", "after traffic stopped")):
        m = ph.get(key)
        if m:
            print(f"  {label:<26} anon {m['anon_mean_mb']:8.1f} MB   "
                  f"cgroup current {m['current_mean_mb']:8.1f} MB")
    if report.get("startup_seconds"):
        print(f"  {'startup to ready':<26} {report['startup_seconds']:8.1f} s")

    base_anon = ph["baseline"]["anon_mean_mb"] if ph.get("baseline") else 0.0

    if report["sweep"]:
        print("\nConcurrency sweep")
        print("-" * 72)
        print(f"  {'c':>5}  {'rps':>8}  {'p95 ms':>8}  {'anon peak':>10}  "
              f"{'delta':>8}  {'MB/rps':>8}  {'err':>5}")
        for row in report["sweep"]:
            load, mem = row["load"], row["mem"]
            delta = mem["anon_max_mb"] - base_anon
            per_rps = mem["anon_max_mb"] / load["rps"] if load["rps"] > 0 else 0.0
            print(f"  {load['concurrency']:>5}  {load['rps']:>8.1f}  {load['p95_ms']:>8.0f}  "
                  f"{mem['anon_max_mb']:>10.1f}  {delta:>8.1f}  {per_rps:>8.2f}  "
                  f"{load['errors']:>5}")
        print(f"\n  {DIM}delta = peak above warm baseline. MB/rps = total anon / achieved rps -")
        print(f"  compare against DRUM, which needs one worker process per concurrent "
              f"request.{NC}")

    leak = report.get("leak")
    if leak:
        soak = report["soak_load"]
        print("\nSoak / leak check")
        print("-" * 72)
        print(f"  requests               {soak['requests']} @ c={soak['concurrency']} "
              f"over {soak['elapsed_s']:.0f}s ({soak['rps']:.1f} rps)")
        print(f"  drift                  {leak['slope_mb_per_1k_requests']:+.3f} MB "
              f"per 1000 requests  ({leak['slope_mb_per_s']:+.4f} MB/s)")
        print(f"  first vs last quarter  {leak['first_quarter_mean_mb']:.1f} -> "
              f"{leak['last_quarter_mean_mb']:.1f} MB  "
              f"({leak['quarter_delta_mb']:+.1f} MB)")

    if ph.get("cooldown") and ph.get("baseline"):
        retained = ph["cooldown"]["anon_mean_mb"] - base_anon
        print(f"\n  retained after traffic  {retained:+.1f} MB vs warm baseline")

    if report.get("kernel_peak_mb"):
        print(f"  kernel high-water mark  {report['kernel_peak_mb']:.1f} MB "
              f"(cgroup memory.peak, whole run)")

    procs = report.get("procs_after_sweep") or []
    if procs:
        print("\nProcesses in container after the sweep")
        print("-" * 72)
        for p in procs[:8]:
            print(f"  pid {p['pid']:>7}  rss {p['rss_mb']:>8.1f} MB  "
                  f"threads {p['threads']:>3}  {p['cmd'][:44]}")
        print(f"  {DIM}per-process RSS double-counts pages shared by fork; the cgroup "
              f"numbers above are authoritative.{NC}")

    stamp = report.get("run_stamp", "")
    print(f"\n  raw samples: {report.get('samples_file')}")
    name = f"report-{report['runner']}-{stamp}.json"
    print(f"  json report: {Path(args.out) / name}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; removing container.")
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
        sys.exit(130)
