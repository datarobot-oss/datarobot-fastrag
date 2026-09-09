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

It also labels two things the numbers cannot show on their own: which adapter
fastrag chose (async hooks await on the event loop; sync hooks run in a
THREAD_POOL_WORKERS-sized pool, which caps concurrency regardless of load), and
whether the container ran emulated - on Apple Silicon rps and latency do not
transfer to production, though the memory figures do.

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
    docker pull --platform linux/amd64 \
        datarobotdev/buzok-genai-custom-model-local-dropin-env:latest
"""

import argparse
import ast
import asyncio
import json
import platform
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
# The base image is published for linux/amd64 only.
DEFAULT_PLATFORM = "linux/amd64"
CONTAINER = "fastrag-memprofile"
MB = 1024.0 * 1024.0

# Hook names fastrag looks for in custom.py (HookName in fastrag/loader.py).
HOOK_NAMES = frozenset(
    {"init", "load_model", "score", "score_unstructured", "chat", "get_supported_llm_models"}
)

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


def warn(msg):
    print(f"  {YELLOW}!{NC} {msg}", flush=True)


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


def fmt_mb(value, width, places=1, sign=""):
    """Right-aligned MB reading, or "n/a" when the sample window was empty.

    A missing reading is not a zero one. summarize_window returns None once the
    sampler stops producing samples (the container died), and printing 0.0 there
    reads as a measurement - 0 MB of anonymous memory never happens.
    """
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{sign}{width}.{places}f}"


def detect_adapter(model_dir):
    """Which adapter fastrag will pick for this model: "async", "sync" or None.

    fastrag/loader.py picks AsyncModelAdapter when any hook is a coroutine function
    and SyncModelAdapter otherwise, and the two have completely different
    concurrency behaviour - so a report that does not say which one ran invites
    being read as a claim about the other. Parsed rather than imported: importing
    custom.py would need the model's own dependencies installed here.

    Returns None when the hooks are not plain module-level defs (a callable class,
    say), because then only fastrag itself can decide.
    """
    try:
        tree = ast.parse((model_dir / "custom.py").read_text())
    except (OSError, SyntaxError):
        return None
    kinds = {
        type(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in HOOK_NAMES
    }
    if not kinds:
        return None
    # loader.py takes the async adapter if *any* hook is async.
    return "async" if ast.AsyncFunctionDef in kinds else "sync"


def adapter_label(adapter, thread_pool_workers, runner="fastrag"):
    """One line naming the adapter and what bounds its concurrency.

    AsyncModelAdapter/SyncModelAdapter are fastrag's own classes, so for the DRUM
    side only the model's hook style is stated.
    """
    if runner != "fastrag":
        return {"async": "async hooks", "sync": "sync hooks"}.get(adapter, "unknown hook style")
    if adapter == "async":
        return "AsyncModelAdapter - hooks await on the event loop, no per-request thread"
    if adapter == "sync":
        pool = (
            f"{thread_pool_workers}-thread pool (THREAD_POOL_WORKERS)"
            if thread_pool_workers
            else "2-thread pool (THREAD_POOL_WORKERS unset, fastrag's default)"
        )
        return f"SyncModelAdapter - hooks run in a {pool}"
    return "unknown - hooks are not plain module-level defs; only fastrag can say"


# ---------------------------------------------------------------------------
# Docker plumbing
# ---------------------------------------------------------------------------


def sh(cmd, check=True, capture=True):
    """Run a command, reporting what it printed if it fails."""
    res = subprocess.run(cmd, check=False, capture_output=capture, text=True)
    if check and res.returncode != 0:
        bad(f"Command failed (exit {res.returncode}): {' '.join(cmd)}")
        for stream in (res.stdout, res.stderr):
            if stream:
                print(stream.rstrip())
        sys.exit(res.returncode)
    return res


def require_docker():
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        bad("Docker daemon is not reachable. Start Docker Desktop and retry.")
        sys.exit(1)


def image_exists(name):
    res = subprocess.run(["docker", "image", "inspect", name], capture_output=True)
    return res.returncode == 0


def build_wheel():
    info("Building wheel (uv build --wheel)...")
    stale = list((REPO_ROOT / "dist").glob("datarobot_fastrag-*.whl"))
    for wheel in stale:
        wheel.unlink()
    if stale:
        info(f"Removed {len(stale)} previously built wheel(s) from dist/")
    sh(["uv", "build", "--wheel", "-q"])
    wheels = sorted((REPO_ROOT / "dist").glob("datarobot_fastrag-*.whl"))
    if not wheels:
        bad("No wheel found in dist/ after build.")
        sys.exit(1)
    ok(f"Wheel: {wheels[-1].name}")


def pull_base_image(base_image, platform):
    info(f"Pulling {base_image} ({platform}) - one-time, several GB...")
    # Inherit stdout so docker's own progress bars stay visible; this takes minutes.
    if subprocess.run(["docker", "pull", "--platform", platform, base_image]).returncode != 0:
        bad("Pull failed. If it was an auth error, run: docker login")
        sys.exit(1)
    ok(f"Pulled {base_image}")


def require_base_image(base_image, platform, do_pull):
    """Fail (or pull) before the caller spends time building anything."""
    if image_exists(base_image):
        return
    if do_pull:
        pull_base_image(base_image, platform)
        return
    bad(f"Base image not present locally: {base_image}")
    print()
    print("    It is a one-time download of several GB. The image ships linux/amd64")
    print("    only, so --platform is required on an arm64 host:")
    print()
    print(f"      docker pull --platform {platform} \\")
    print(f"          {base_image}")
    print()
    print("    Or let this script do it:")
    print('      make mem-profile ARGS="--pull"')
    print()
    print("    A pull that prints 'no matching manifest for linux/arm64/v8' was missing")
    print("    the --platform flag. One that is still running has not finished yet -")
    print(f"    check with: docker image ls {base_image.split(':')[0]}")
    sys.exit(1)


def build_image(base_image, tag, platform):
    info(f"Building image {tag} from {base_image} ({platform})...")
    # Without --platform the daemon resolves FROM for the host arch, and the base
    # image has no arm64 entry, so the build dies on Apple Silicon before it starts.
    sh(["docker", "build", "--platform", platform,
        "-f", "Dockerfile.local-test", "-t", tag, "--quiet", "."])
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
    # Governs SyncModelAdapter's thread pool (server.py passes it to HookLoader),
    # so it, not MAX_WORKERS, is the concurrency ceiling for a sync model.
    if args.thread_pool_workers:
        env["THREAD_POOL_WORKERS"] = str(args.thread_pool_workers)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    run_cmd = [
        "docker", "run", "-d",
        "--platform", args.platform,
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


def container_state():
    """(running, oom_killed, exit_code); running=False if the container is gone."""
    res = subprocess.run(
        ["docker", "inspect", "-f",
         "{{.State.Running}} {{.State.OOMKilled}} {{.State.ExitCode}}", CONTAINER],
        capture_output=True, text=True,
    )
    parts = res.stdout.split()
    if len(parts) != 3:
        return (False, False, None)
    return (parts[0] == "true", parts[1] == "true", int(parts[2]))


def normalise_arch(name):
    """Collapse the aliases docker, uname and platform.machine each prefer."""
    if not name:
        return None
    name = name.strip().lower()
    if name in {"x86_64", "amd64", "x86-64"}:
        return "x86_64"
    if name in {"arm64", "aarch64"}:
        return "arm64"
    return name


def arch_info():
    """Container arch vs host arch, and whether this run is emulated.

    Recorded because absolute rps and latency from an emulated run do not transfer
    to production, and a report that does not say so will eventually be quoted as
    if it did. Memory figures survive emulation; throughput does not.
    """
    res = subprocess.run(
        ["docker", "exec", CONTAINER, "uname", "-m"], capture_output=True, text=True
    )
    container = normalise_arch(res.stdout)
    host = normalise_arch(platform.machine())
    return {
        "container": container,
        "host": host,
        "emulated": bool(container and host and container != host),
    }


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
        if not container_state()[0]:
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
    p.add_argument("--platform", default=DEFAULT_PLATFORM,
                   help=f"Container platform (default: {DEFAULT_PLATFORM}, the only arch the "
                        f"base image ships). On arm64 this runs under emulation.")
    p.add_argument("--no-build", action="store_true", help="Skip wheel + image build")
    p.add_argument("--pull", action="store_true",
                   help="Pull the base image if it is missing instead of failing")
    p.add_argument("--memory", default="2g", help="Container memory limit (default: 2g)")
    p.add_argument("--max-workers", type=int, default=1, help="MAX_WORKERS (default: 1)")
    p.add_argument("--thread-pool-workers", type=int, default=0,
                   help="Set THREAD_POOL_WORKERS, which caps concurrency for a sync model "
                        "(0 = leave unset, fastrag defaults to 2)")
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
    adapter = detect_adapter(model_dir)

    if args.no_build:
        if not image_exists(args.image):
            bad(f"Image {args.image} not found and --no-build was passed.")
            print("    Drop --no-build to build it, or pass --image with one that exists.")
            sys.exit(1)
    else:
        require_base_image(args.base_image, args.platform, args.pull)

    print()
    print(f"{args.runner} memory profile")
    print("=" * 72)
    print(f"  model-dir   {model_dir}")
    print(f"  image       {args.image}")
    print(f"  mem limit   {args.memory}   MAX_WORKERS={args.max_workers}"
          f"   MALLOC_ARENA_MAX={args.malloc_arena_max or 'unset'}")
    print(f"  adapter     {adapter_label(adapter, args.thread_pool_workers, args.runner)}")
    print()
    if adapter == "sync" and args.runner == "fastrag":
        warn("Sync model: throughput is capped by the thread pool, not the event loop, "
             "so a flat memory curve here says little about the async path.")

    if not args.no_build:
        build_wheel()
        build_image(args.base_image, args.image, args.platform)

    sampler = None
    report = {
        "runner": args.runner,
        "model_dir": str(model_dir),
        "adapter": adapter,
        "memory_limit": args.memory,
        "max_workers": args.max_workers,
        "malloc_arena_max": args.malloc_arena_max or None,
        "thread_pool_workers": args.thread_pool_workers or None,
        "phases": {},
        "sweep": [],
    }

    try:
        start_container(args, model_dir)
        report["startup_seconds"] = wait_ready(base_url, args.ready_timeout)

        arch = arch_info()
        report["arch"] = arch
        if arch["emulated"]:
            warn(f"Container is {arch['container']} on an {arch['host']} host: running under "
                 f"emulation. Memory figures still hold; rps and latency do not transfer "
                 f"to production.")

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
        warm = report["phases"]["baseline"]
        base_anon = warm["anon_mean_mb"] if warm else None

        # 4. concurrency sweep
        print()
        info(f"concurrency sweep: {levels}  ({args.phase_seconds}s each)")
        died_at = None
        for c in levels:
            load = asyncio.run(drive(chat_url, payload, c, duration_s=args.phase_seconds))
            mem = summarize_window(sampler.window(load["t0"], load["t1"]))
            row = {"load": load, "mem": mem}
            report["sweep"].append(row)
            anon_peak = mem["anon_max_mb"] if mem else None
            delta = None if anon_peak is None or base_anon is None else anon_peak - base_anon
            print(f"    c={c:<4} {load['rps']:7.1f} rps  p95={load['p95_ms']:6.0f}ms  "
                  f"anon peak {fmt_mb(anon_peak, 7)} MB  ({fmt_mb(delta, 6, sign='+')})  "
                  f"err={load['errors']}")
            if load["errors"]:
                print(f"         {RED}error sample:{NC} {load['error_sample']}")

            # Death here is the headline result, not an error to sample past: the
            # remaining levels would hammer nothing and report n/a rows.
            running, oom_killed, exit_code = container_state()
            if not running:
                died_at = c
                report["died_at_concurrency"] = c
                report["oom_killed"] = oom_killed
                reason = "OOM-killed" if oom_killed else f"exit code {exit_code}"
                bad(f"Container died at c={c} ({reason}) - stopping the sweep. Logs:")
                print(logs(tail=40))
                break
            time.sleep(args.settle)

        report["procs_after_sweep"] = proc_snapshot()

        # 5. soak - leak detection
        if args.soak_requests and died_at is None:
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
        if died_at is None:
            print()
            cool = idle(args.cooldown_seconds, "cooldown")
            report["phases"]["cooldown"] = summarize_window(
                sampler.window(cool["t0"], cool["t1"])
            )
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


def arch_summary(report):
    """Short "x86_64 emulated on arm64" tag for the header, or "" if unknown."""
    arch = report.get("arch") or {}
    if not arch.get("container"):
        return ""
    if arch.get("emulated"):
        return f"{arch['container']} emulated on {arch['host']}"
    return arch["container"]


def print_report(report, args):
    ph = report["phases"]
    print()
    print("=" * 72)
    title = (f"{report['runner']}  |  limit {report['memory_limit']}  |  "
             f"MAX_WORKERS={report['max_workers']}")
    tag = arch_summary(report)
    print(f"{title}  |  {tag}" if tag else title)
    print("=" * 72)
    label = adapter_label(
        report.get("adapter"), report.get("thread_pool_workers"), report["runner"]
    )
    print(f"  {DIM}adapter: {label}{NC}")
    if (report.get("arch") or {}).get("emulated"):
        print(f"  {YELLOW}Emulated run: trust the memory columns and the shape of the "
              f"curve, not absolute rps or latency.{NC}")

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

    base_anon = ph["baseline"]["anon_mean_mb"] if ph.get("baseline") else None

    if report["sweep"]:
        print("\nConcurrency sweep")
        print("-" * 72)
        print(f"  {'c':>5}  {'rps':>8}  {'p95 ms':>8}  {'anon peak':>10}  "
              f"{'delta':>8}  {'MB/rps':>8}  {'err':>5}")
        for row in report["sweep"]:
            load, mem = row["load"], row["mem"]
            anon_peak = mem["anon_max_mb"] if mem else None
            delta = None if anon_peak is None or base_anon is None else anon_peak - base_anon
            per_rps = anon_peak / load["rps"] if anon_peak is not None and load["rps"] > 0 else None
            print(f"  {load['concurrency']:>5}  {load['rps']:>8.1f}  {load['p95_ms']:>8.0f}  "
                  f"{fmt_mb(anon_peak, 10)}  {fmt_mb(delta, 8)}  {fmt_mb(per_rps, 8, places=2)}  "
                  f"{load['errors']:>5}")
        print(f"\n  {DIM}delta = peak above warm baseline. MB/rps = total anon / achieved rps.")
        print("  For the other side of the comparison run --runner drum against a sync")
        print("  model, which holds a worker for each in-flight request instead of")
        print(f"  awaiting; that side is not measured yet.{NC}")

    if report.get("died_at_concurrency") is not None:
        reason = "OOM-killed" if report.get("oom_killed") else "exited"
        bad(f"Container {reason} at c={report['died_at_concurrency']} - "
            f"levels above it were not measured.")

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
        print(f"  {DIM}RSS counts pages shared between processes once per process, and")
        print("  includes mapped file pages (interpreter, shared libs) the cgroup may")
        print("  charge elsewhere - so it can exceed the cgroup total. The cgroup")
        print(f"  numbers above are authoritative.{NC}")

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
