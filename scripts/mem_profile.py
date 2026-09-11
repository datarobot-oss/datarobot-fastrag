#!/usr/bin/env python3
"""
Local memory profiler for a containerised custom model runner (fastrag or DRUM).

Runs one container under a fixed memory limit, samples the container's cgroup
memory while driving chat traffic at it, and reports:

  cold baseline   memory right after load_model, zero traffic  -> OOM/migration risk
  sweep           peak memory at each concurrency level        -> where the curve bends
  MB per inflight peak above baseline / in-flight requests     -> marginal cost vs DRUM
  core-ms/req     CPU time charged per request                 -> efficiency at 1 CPU
  leak slope      memory drift over a long soak                -> per-request leaks
  retained        memory left over after traffic stops         -> what never comes back

It also labels two things the numbers cannot show on their own: which adapter
fastrag chose (async hooks await on the event loop; sync hooks run in a
THREAD_POOL_WORKERS-sized pool, which caps concurrency regardless of load), and
whether the container ran emulated - on Apple Silicon rps and latency do not
transfer to production, though the memory figures do.

Memory is read from the container's own cgroup (v2 preferred, v1 fallback), which
is the number the OOM killer acts on. Four series are recorded:

  anon      anonymous (heap) pages - the fairest cross-runtime metric
  current   everything charged to the cgroup, incl. page cache - the limit's view
  cpu       usage_usec from cpu.stat, so a phase can be priced in core-seconds
  throttle  nr_throttled/throttled_usec - why latency cliffs at a --cpus limit

Pass --cpus to match the CPU allowance of the resource bundle being claimed.
Without it the container gets every host core, and no rps or latency figure from
the run says anything about a 1-CPU deployment.

Usage:
    # full run: build wheel + image, profile, report
    uv run scripts/mem_profile.py --model-dir tests/benchmark_model_rag

    # reuse an already-built image, quick pass
    uv run scripts/mem_profile.py --no-build --phase-seconds 8 --soak-requests 2000

    # one side of a 1 CPU / 1 GB bundle comparison
    uv run scripts/mem_profile.py --no-build --cpus 1 --memory 1g \
        --model-dir tests/benchmark_model_rag --payload payloads/rag.json

    # DRUM side of the same comparison (same image, sync model). MAX_WORKERS is
    # DRUM's concurrency ceiling, not just a worker count - see start_container.
    uv run scripts/mem_profile.py --runner drum --no-build --cpus 1 --memory 1g \
        --model-dir tests/benchmark_model_rag_sync --max-workers 1

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

# One long-lived python3 process rather than a shell loop. The shell version
# forked cat/grep/cut six times a sample, which under emulation cost 0.56 of a
# core and produced 16 CPU-throttle events in six idle seconds at --cpus 1 - it
# perturbed the system under test, not just the reading. This costs 0.05 cores
# and throttles never. It does hold ~10 MB more anon than the shell did, which is
# charged to the cgroup being measured, so its own RSS is reported alongside the
# baselines rather than quietly folded into them.
#
# The marker in the first line lands in the process cmdline, so pkill -f can find
# it when --keep leaves the container running.
SAMPLER_PY = r"""# memprofile-sampler
import sys, time

def read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""

def field(text, key, divisor=1):
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == key:
            try:
                return int(parts[1]) // divisor
            except ValueError:
                return 0
    return 0

def num(path, divisor=1):
    try:
        return int(read(path).strip() or 0) // divisor
    except ValueError:
        return 0

interval = float(sys.argv[1])
v2 = bool(read("/sys/fs/cgroup/memory.current"))
cpuacct = "/sys/fs/cgroup/cpu,cpuacct"
if not v2 and not read(cpuacct + "/cpuacct.usage"):
    cpuacct = "/sys/fs/cgroup/cpuacct"

while True:
    if v2:
        cur = num("/sys/fs/cgroup/memory.current")
        anon = field(read("/sys/fs/cgroup/memory.stat"), "anon")
        peak = num("/sys/fs/cgroup/memory.peak")
        cpu = read("/sys/fs/cgroup/cpu.stat")
        usage = field(cpu, "usage_usec")
        throttled_n = field(cpu, "nr_throttled")
        throttled_us = field(cpu, "throttled_usec")
    else:
        # v1 splits these across controllers and counts CPU in nanoseconds.
        cur = num("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        anon = field(read("/sys/fs/cgroup/memory/memory.stat"), "total_rss")
        peak = num("/sys/fs/cgroup/memory/memory.max_usage_in_bytes")
        usage = num(cpuacct + "/cpuacct.usage", 1000)
        cpu = read(cpuacct + "/cpu.stat")
        throttled_n = field(cpu, "nr_throttled")
        throttled_us = field(cpu, "throttled_time", 1000)
    print(cur, anon, peak, usage, throttled_n, throttled_us, flush=True)
    time.sleep(interval)
"""

SAMPLER_MARKER = "memprofile-sampler"

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
    """Streams cgroup memory + CPU samples out of the container via one long-lived exec."""

    #: Field order of every entry in .samples, and of the TSV written at the end.
    FIELDS = ("host_time", "current_bytes", "anon_bytes", "peak_bytes",
              "cpu_usage_usec", "nr_throttled", "throttled_usec")

    def __init__(self, container, interval=0.25):
        self.container = container
        self.interval = interval
        self.samples = []  # tuples in Sampler.FIELDS order
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        # -S -E: no site imports, no PYTHON* env from the image - smaller and
        # faster to start. Output is flushed per line by the program itself.
        self._proc = subprocess.Popen(
            ["docker", "exec", self.container,
             "python3", "-S", "-E", "-c", SAMPLER_PY, str(self.interval)],
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
                "cgroup sampler produced no samples; is /sys/fs/cgroup readable in the "
                "container, and is python3 on its PATH?"
            )

    def _read(self):
        for line in self._proc.stdout:
            if self._stop.is_set():
                return
            parts = line.split()
            if len(parts) != len(self.FIELDS) - 1:
                continue
            try:
                values = tuple(int(p) for p in parts)
            except ValueError:
                continue
            self.samples.append((time.time(),) + values)

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
            ["docker", "exec", self.container, "pkill", "-f", SAMPLER_MARKER],
            capture_output=True,
        )

    def window(self, t0, t1):
        return [s for s in self.samples if t0 <= s[0] <= t1]

    def peak_bytes(self):
        """Kernel-tracked high-water mark, immune to sampling gaps (0 if unsupported)."""
        return max((s[3] for s in self.samples), default=0)


def summarize_window(samples):
    """Memory and CPU summary for one phase window.

    The memory fields are point-in-time gauges, so they are averaged. CPU comes
    from counters, so it is a first-to-last delta over the window's own wall
    clock - which needs two samples, and is omitted rather than guessed at when
    the window is shorter than that.
    """
    if not samples:
        return None
    anon = [s[2] / MB for s in samples]
    cur = [s[1] / MB for s in samples]
    out = {
        "n_samples": len(samples),
        "anon_mean_mb": statistics.fmean(anon),
        "anon_max_mb": max(anon),
        "anon_min_mb": min(anon),
        "current_mean_mb": statistics.fmean(cur),
        "current_max_mb": max(cur),
    }
    if len(samples) >= 2:
        first, last = samples[0], samples[-1]
        wall = last[0] - first[0]
        cpu_s = (last[4] - first[4]) / 1e6
        out.update({
            "wall_seconds": wall,
            "cpu_seconds": cpu_s,
            # Cores consumed: 1.00 is one core saturated for the whole window, so
            # it reads directly against the --cpus limit.
            "cpu_cores_mean": cpu_s / wall if wall > 0 else None,
            "throttled_periods": last[5] - first[5],
            "throttled_seconds": (last[6] - first[6]) / 1e6,
        })
    return out


def derive_row(load, mem, base_anon):
    """Per-level metrics the raw columns do not give directly.

    mb_per_inflight is the marginal cost of holding one more request in flight -
    the figure that actually separates an event loop from a worker per request.
    Total anon over achieved rps was the earlier metric; it is dominated by the
    fixed baseline, so it improves with concurrency for any runner and cannot be
    compared across two.
    """
    anon_peak = mem["anon_max_mb"] if mem else None
    concurrency = load["concurrency"]
    delta = None if anon_peak is None or base_anon is None else anon_peak - base_anon
    cpu_s = (mem or {}).get("cpu_seconds")
    requests = load["requests"]
    return {
        "anon_peak_mb": anon_peak,
        "delta_mb": delta,
        "mb_per_inflight": delta / concurrency if delta is not None and concurrency else None,
        "cores_mean": (mem or {}).get("cpu_cores_mean"),
        "core_ms_per_request": (
            cpu_s / requests * 1000.0 if cpu_s is not None and requests else None
        ),
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


def runner_cmd(runner, target_type, max_workers, log_level):
    """(entrypoint, argv) for the runner, mirroring /opt/code/start_server.sh.

    Verified end to end against datarobot-drum 1.17.17 in the base image: it
    registers both /chat/completions and /v1/chat/completions, so one chat URL
    drives either runner, and a profiling run completes.

    --max-workers is passed on the command line rather than left to the
    MAX_WORKERS env var so the concurrency ceiling is visible in the printed
    docker command. It is a ceiling and not merely a worker count: DRUM ends up
    at app.run(threaded=False, processes=max_workers), which serves exactly
    max_workers requests at a time and forks per request above 1. At 1 the
    measured result is a flat 1/latency at every concurrency level, with p95
    growing linearly as the queue does.

    --logging-level is passed because DRUM is otherwise completely silent -
    docker logs returns zero bytes - while fastrag's uvicorn logs every request.
    That is both a missing diagnostic and a real CPU asymmetry at 1 CPU, so the
    level is an explicit, recorded choice on both sides.
    """
    if runner == "fastrag":
        return (
            ["fastrag"],
            ["server", "--code-dir", "/opt/model", "--address", "0.0.0.0:8080"],
        )
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
            "--max-workers",
            str(max_workers),
            "--logging-level",
            log_level,
        ],
    )


def start_container(args, model_dir):
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    entrypoint, cmd = runner_cmd(
        args.runner, args.target_type, args.max_workers, args.log_level
    )

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
    # A CPU quota is what makes the run represent a resource bundle rather than
    # this laptop. Left off by default because it changes every throughput
    # figure, so it has to be an explicit choice.
    if args.cpus:
        run_cmd += ["--cpus", str(args.cpus)]
    for k, v in env.items():
        run_cmd += ["-e", f"{k}={v}"]
    run_cmd += [args.image] + cmd

    info(f"Starting {args.runner} container (limit={args.memory}, "
         f"cpus={args.cpus or 'unlimited'}, MAX_WORKERS={args.max_workers})")
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


def cpu_info():
    """What the container sees vs what the quota actually allows.

    A CFS quota (--cpus) is invisible to the process: nproc and os.cpu_count()
    still report every host core, so glibc arena counts (8 x ncores), thread
    pools and tokenizer pools all size themselves for a machine the container
    cannot use. That inflates memory, not just CPU, which is why it is recorded
    next to the memory figures rather than filed under performance.
    """
    res = subprocess.run(
        ["docker", "exec", CONTAINER, "sh", "-c",
         "nproc 2>/dev/null || echo ?; cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unknown"],
        capture_output=True, text=True,
    )
    lines = res.stdout.strip().splitlines()
    visible = None
    if lines:
        try:
            visible = int(lines[0].strip())
        except ValueError:
            pass
    cpu_max = lines[1].strip() if len(lines) > 1 else "unknown"
    quota_cores = None
    parts = cpu_max.split()
    if len(parts) == 2 and parts[0] != "max":
        try:
            quota, period = int(parts[0]), int(parts[1])
            quota_cores = quota / period if period else None
        except ValueError:
            pass
    return {"nproc_visible": visible, "cpu_max": cpu_max, "quota_cores": quota_cores}


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
    """Container log, or all of it when tail is None."""
    res = subprocess.run(
        ["docker", "logs", "--tail", "all" if tail is None else str(tail), CONTAINER],
        capture_output=True, text=True,
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


def snapshot_during(delay):
    """Schedule a proc snapshot `delay` seconds from now; returns (box, thread).

    Taken while traffic is still flowing, because the interesting processes are
    gone by the time a phase ends: DRUM forks per request above MAX_WORKERS=1 and
    reaps each fork on completion, so a post-phase snapshot shows none of them.

    It costs a docker exec and a walk of /proc inside the container, which is
    charged to the same cgroup being measured - one snapshot per level, so a
    blip, but it is not free at --cpus 1.
    """
    box = {}

    def run():
        time.sleep(delay)
        box["procs"] = proc_snapshot()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return box, thread


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


def fit_sweep(sweep):
    """Marginal MB per in-flight request, fitted across the whole sweep.

    Dividing one level's delta by its own concurrency (the MB/inflt column)
    charges that level for every megabyte the baseline happened to have drifted
    by the time it ran, which at c=1 swamps the per-request cost completely. A
    fit across levels separates the two: the slope is the marginal cost of one
    more in-flight request, the intercept is the fixed footprint. This is the
    number to carry into a cross-runner comparison.

    Levels that returned errors are dropped - a level that was shedding requests
    was not holding them in flight, so it does not belong on the line.
    """
    points = [
        (row["load"]["concurrency"], row["mem"]["anon_max_mb"])
        for row in sweep
        if row.get("mem") and not row["load"]["errors"]
    ]
    if len(points) < 3:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    slope = linreg_slope(xs, ys)
    return {
        "slope_mb_per_inflight": slope,
        "intercept_mb": statistics.fmean(ys) - slope * statistics.fmean(xs),
        "n_points": len(points),
        "levels": xs,
    }


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
    p.add_argument("--cpus", default=None,
                   help="Container CPU limit, e.g. 1 or 1.5 (default: unlimited). Required for "
                        "any figure quoted as representing an N-CPU resource bundle.")
    p.add_argument("--max-workers", type=int, default=1,
                   help="MAX_WORKERS (default: 1). For DRUM this is the number of requests it "
                        "will serve at a time, so it caps rps at max_workers/latency.")
    p.add_argument("--thread-pool-workers", type=int, default=0,
                   help="Set THREAD_POOL_WORKERS, which caps concurrency for a sync model "
                        "(0 = leave unset, fastrag defaults to 2)")
    p.add_argument("--malloc-arena-max", type=int, default=0,
                   help="Set MALLOC_ARENA_MAX (0 = leave unset, matching production)")
    p.add_argument("--payload", metavar="FILE",
                   help="JSON file with the chat request body (default: a one-message 'hello'). "
                        "A trivial prompt hides per-request memory; use a realistic RAG-sized "
                        "body for anything that will be quoted as a per-request cost.")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"],
                   help="Runner log level (default: info). DRUM logs nothing at all unless "
                        "this is set, while fastrag's uvicorn logs every request at info - "
                        "so leave both here to keep the two sides symmetric.")
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


def preflight_warnings(args, adapter):
    """Everything a reader of the report would otherwise have to already know.

    Each of these silently changes what the numbers mean, so they are said before
    the run rather than left for whoever quotes the output later.
    """
    if adapter == "sync" and args.runner == "fastrag":
        warn("Sync model: throughput is capped by the thread pool, not the event loop, "
             "so a flat memory curve here says little about the async path.")
        if not args.thread_pool_workers:
            warn("THREAD_POOL_WORKERS is unset, so fastrag defaults to 2 threads. Against a "
                 "DRUM run with a higher MAX_WORKERS that is not a like-for-like comparison - "
                 "set both to the same number.")
    if args.runner == "drum" and args.max_workers <= 1:
        warn("DRUM at MAX_WORKERS=1 runs Flask with threaded=False, processes=1: one request "
             "at a time, so rps is capped at 1/latency at every concurrency level. That is "
             "a real production ceiling, not a harness artefact - but use the MAX_WORKERS the "
             "target resource bundle actually passes.")
    if not args.cpus:
        warn("No --cpus limit: the container may use every host core. Memory figures still "
             "hold, but no rps or latency number from this run describes an N-CPU bundle.")


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
    if args.payload:
        payload_path = Path(args.payload)
        if not payload_path.is_absolute():
            payload_path = REPO_ROOT / payload_path
        try:
            payload = json.loads(payload_path.read_text())
        except (OSError, ValueError) as exc:
            bad(f"Could not read --payload {payload_path}: {exc}")
            sys.exit(1)
    else:
        payload = {
            "model": "memprofile",
            "messages": [{"role": "user", "content": "hello"}],
        }
    payload_bytes = len(json.dumps(payload).encode())
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
    print(f"  limits      mem {args.memory}   cpus {args.cpus or 'unlimited'}"
          f"   MAX_WORKERS={args.max_workers}"
          f"   MALLOC_ARENA_MAX={args.malloc_arena_max or 'unset'}")
    print(f"  payload     {args.payload or 'built-in one-message hello'} ({payload_bytes} B)")
    print(f"  adapter     {adapter_label(adapter, args.thread_pool_workers, args.runner)}")
    print()
    preflight_warnings(args, adapter)

    if not args.no_build:
        build_wheel()
        build_image(args.base_image, args.image, args.platform)

    sampler = None
    report = {
        "runner": args.runner,
        "model_dir": str(model_dir),
        "adapter": adapter,
        "memory_limit": args.memory,
        "cpu_limit": args.cpus,
        "payload_file": args.payload,
        "payload_bytes": payload_bytes,
        "log_level": args.log_level,
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

        cpu = cpu_info()
        report["cpu"] = cpu
        quota, visible = cpu["quota_cores"], cpu["nproc_visible"]
        if quota and visible and visible > quota:
            warn(f"Container sees {cpu['nproc_visible']} cores but may only use "
                 f"{cpu['quota_cores']:g}: the CFS quota is invisible to the process, so "
                 f"arena counts and thread pools are sized for {cpu['nproc_visible']} cores. "
                 f"Production behaves the same way; try --malloc-arena-max to price it.")

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
            # Fires mid-phase, while the processes serving the load still exist.
            proc_box, proc_thread = snapshot_during(args.phase_seconds * 0.6)
            load = asyncio.run(drive(chat_url, payload, c, duration_s=args.phase_seconds))
            proc_thread.join(timeout=30)
            mem = summarize_window(sampler.window(load["t0"], load["t1"]))
            derived = derive_row(load, mem, base_anon)
            row = {
                "load": load,
                "mem": mem,
                "derived": derived,
                "procs_at_load": proc_box.get("procs"),
            }
            report["sweep"].append(row)
            print(f"    c={c:<4} {load['rps']:7.1f} rps  p95={load['p95_ms']:6.0f}ms  "
                  f"anon peak {fmt_mb(derived['anon_peak_mb'], 7)} MB  "
                  f"({fmt_mb(derived['delta_mb'], 6, sign='+')})  "
                  f"cores {fmt_mb(derived['cores_mean'], 5, places=2)}  "
                  f"err={load['errors']}")
            if load["errors"]:
                print(f"         {RED}error sample:{NC} {load['error_sample']}")
            if mem and mem.get("throttled_periods"):
                print(f"         {YELLOW}CPU-throttled {mem['throttled_periods']} periods, "
                      f"{mem['throttled_seconds']:.2f}s{NC}")

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
        report["sweep_fit"] = fit_sweep(report["sweep"])

        # 5. soak - leak detection
        if args.soak_requests and died_at is None:
            print()
            info(f"soak: {args.soak_requests} requests @ c={args.soak_concurrency}")
            soak = asyncio.run(
                drive(chat_url, payload, args.soak_concurrency, requests=args.soak_requests)
            )
            report["phases"]["soak"] = summarize_window(sampler.window(soak["t0"], soak["t1"]))
            report["soak_load"] = soak
            report["soak_derived"] = derive_row(soak, report["phases"]["soak"], base_anon)
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
                fh.write("\t".join(Sampler.FIELDS) + "\n")
                for s in sampler.samples:
                    fh.write("\t".join(str(x) for x in s) + "\n")
            report["samples_file"] = str(tsv)
            report["run_stamp"] = stamp
            # Kept unconditionally: a throughput cliff or a stray disconnect is
            # explained in here, and by the time anyone asks the container is gone.
            container_log = logs(tail=None)
            if container_log.strip():
                log_file = outdir / f"logs-{args.runner}-{stamp}.txt"
                log_file.write_text(container_log)
                report["logs_file"] = str(log_file)
            else:
                # Silence is a finding, not a non-event: DRUM emits nothing
                # whatsoever without --logging-level, and a report with no log to
                # explain a latency cliff should say why there is none.
                report["logs_empty"] = True
            report_json = json.dumps(report, indent=2)
            (outdir / f"report-{args.runner}-{stamp}.json").write_text(report_json)
            (outdir / f"report-{args.runner}-latest.json").write_text(report_json)
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


def print_sweep_table(report, base_anon):
    """The concurrency sweep, plus the throttling that explains its shape."""
    if not report["sweep"]:
        return
    print("\nConcurrency sweep")
    print("-" * 94)
    print(f"  {'c':>5}  {'rps':>8}  {'p95 ms':>8}  {'anon peak':>10}  {'delta':>8}  "
          f"{'MB/inflt':>9}  {'cores':>6}  {'core-ms/req':>11}  {'err':>5}")
    for row in report["sweep"]:
        load = row["load"]
        d = row.get("derived") or derive_row(load, row["mem"], base_anon)
        print(f"  {load['concurrency']:>5}  {load['rps']:>8.1f}  {load['p95_ms']:>8.0f}  "
              f"{fmt_mb(d['anon_peak_mb'], 10)}  {fmt_mb(d['delta_mb'], 8)}  "
              f"{fmt_mb(d['mb_per_inflight'], 9, places=3)}  "
              f"{fmt_mb(d['cores_mean'], 6, places=2)}  "
              f"{fmt_mb(d['core_ms_per_request'], 11, places=1)}  "
              f"{load['errors']:>5}")
    print(f"\n  {DIM}delta = peak anon above the warm baseline.")
    print("  MB/inflt = delta / concurrency: the marginal cost of one more request in")
    print("  flight, which is the figure to compare across runners. Total anon over rps")
    print("  is not - the fixed baseline dominates it and it falls with concurrency for")
    print("  any runner. cores = mean cores consumed, so 1.00 saturates --cpus 1.")
    print(f"  core-ms/req = CPU time charged per request.{NC}")

    fit = report.get("sweep_fit")
    if fit:
        print(f"\n  marginal cost  {fit['slope_mb_per_inflight']:+.3f} MB per in-flight "
              f"request, fixed {fit['intercept_mb']:.1f} MB")
        print(f"  {DIM}Fitted over {fit['n_points']} error-free levels {fit['levels']}. Prefer "
              f"this to any single row's")
        print(f"  MB/inflt: at low concurrency that column is mostly baseline drift.{NC}")

    throttled = [
        (row["load"]["concurrency"], row["mem"])
        for row in report["sweep"]
        if row.get("mem") and row["mem"].get("throttled_periods")
    ]
    if not throttled:
        return
    print("\nCPU throttling (the cpu.max quota was hit)")
    print("-" * 94)
    for concurrency, mem in throttled:
        share = (
            mem["throttled_seconds"] / mem["wall_seconds"] * 100
            if mem.get("wall_seconds") else None
        )
        tail = f"  ({share:.0f}% of the phase)" if share is not None else ""
        print(f"  c={concurrency:<5} {mem['throttled_periods']:>6} periods  "
              f"{mem['throttled_seconds']:>7.2f}s{tail}")
    print(f"  {DIM}Throttling, not the runner, is what bent the latency curve here.{NC}")


def sampler_rss_mb(report):
    """RSS of the in-container sampler, so its charge can be netted out.

    It runs inside the cgroup under measurement, so every memory figure includes
    it. RSS slightly overstates the cgroup charge (mapped libraries are shared),
    which makes it a conservative correction rather than a precise one.
    """
    snapshots = [report.get("procs_after_sweep") or []]
    snapshots += [row.get("procs_at_load") or [] for row in report.get("sweep") or []]
    for procs in snapshots:
        for row in procs:
            if SAMPLER_MARKER in row.get("cmd", ""):
                return row["rss_mb"]
    return None


def proc_totals(procs):
    """(process count, total threads) for one snapshot."""
    threads = sum(int(r["threads"]) for r in procs if str(r["threads"]).isdigit())
    return len(procs), threads


def print_process_table(report):
    """Process and thread counts under load, then what was left afterwards.

    The under-load rows are the point: they separate a runner that awaits from one
    that dedicates a worker, or a whole forked interpreter, to each request.
    """
    per_level = [row for row in report["sweep"] if row.get("procs_at_load")]
    after = report.get("procs_after_sweep") or []
    if not per_level and not after:
        return

    print("\nProcesses and threads")
    print("-" * 94)
    if per_level:
        print(f"  {'level':<12} {'procs':>6} {'threads':>8}   busiest process")
        for row in per_level:
            procs = row["procs_at_load"]
            n, threads = proc_totals(procs)
            print(f"  c={row['load']['concurrency']:<10} {n:>6} {threads:>8}   "
                  f"rss {procs[0]['rss_mb']:.0f} MB, {procs[0]['threads']} threads")
        print(f"  {DIM}Sampled mid-phase, under load. A worker-per-request runner shows its"
              f" cost here.{NC}")

    if after:
        n, threads = proc_totals(after)
        print(f"\n  after the sweep: {n} processes, {threads} threads")
        for r in after[:6]:
            print(f"    pid {r['pid']:>7}  rss {r['rss_mb']:>8.1f} MB  "
                  f"threads {r['threads']:>3}  {r['cmd'][:40]}")
        print(f"  {DIM}RSS counts pages shared between processes once per process, and")
        print("  includes mapped file pages (interpreter, shared libs) the cgroup may")
        print("  charge elsewhere - so it can exceed the cgroup total. The cgroup")
        print(f"  numbers above are authoritative.{NC}")


def print_report(report, args):
    ph = report["phases"]
    print()
    print("=" * 72)
    title = (f"{report['runner']}  |  mem {report['memory_limit']}  |  "
             f"cpus {report.get('cpu_limit') or 'unlimited'}  |  "
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
    cpu = report.get("cpu") or {}
    if cpu.get("nproc_visible"):
        quota = cpu.get("quota_cores")
        allowed = f"{quota:g} core(s) by quota" if quota else "no quota (all of them)"
        print(f"  {DIM}cpu: container sees {cpu['nproc_visible']} cores, may use {allowed}{NC}")
    if report.get("payload_bytes"):
        origin = report.get("payload_file") or "built-in hello"
        print(f"  {DIM}payload: {report['payload_bytes']} B ({origin}){NC}")

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
    overhead = sampler_rss_mb(report)
    if overhead:
        print(f"  {DIM}includes the in-container sampler, {overhead:.1f} MB RSS - subtract it "
              f"before sizing a bundle{NC}")

    base_anon = ph["baseline"]["anon_mean_mb"] if ph.get("baseline") else None

    print_sweep_table(report, base_anon)

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
        sd = report.get("soak_derived") or {}
        if sd.get("cores_mean") is not None:
            print(f"  cpu                    {sd['cores_mean']:.2f} cores mean, "
                  f"{fmt_mb(sd['core_ms_per_request'], 1, places=1).strip()} core-ms/request")
        soak_mem = ph.get("soak") or {}
        if soak_mem.get("throttled_periods"):
            print(f"  throttling             {soak_mem['throttled_periods']} periods, "
                  f"{soak_mem['throttled_seconds']:.2f}s")

    if ph.get("cooldown") and ph.get("baseline"):
        retained = ph["cooldown"]["anon_mean_mb"] - base_anon
        print(f"\n  retained after traffic  {retained:+.1f} MB vs warm baseline")

    if report.get("kernel_peak_mb"):
        print(f"  kernel high-water mark  {report['kernel_peak_mb']:.1f} MB "
              f"(cgroup memory.peak, whole run)")

    print_process_table(report)

    stamp = report.get("run_stamp", "")
    print(f"\n  raw samples: {report.get('samples_file')}")
    name = f"report-{report['runner']}-{stamp}.json"
    print(f"  json report: {Path(args.out) / name}")
    if report.get("logs_file"):
        print(f"  container log: {report['logs_file']}")
    elif report.get("logs_empty"):
        warn(f"The container logged nothing at all (--log-level "
             f"{report.get('log_level', '?')}), so there is no log to explain this run.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; removing container.")
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
        sys.exit(130)
