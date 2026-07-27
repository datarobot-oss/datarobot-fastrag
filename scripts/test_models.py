#!/usr/bin/env python3
"""
Custom model testing and load testing tool for fastrag execution environments.

correctness mode
    For each model: run the test on its CURRENT env (baseline), then create a new
    version pointing at the fastrag env and run the same test. Reports pass/fail
    and timing for both. Always establishes a passing baseline before testing fastrag.

load mode
    Send N concurrent requests to an existing deployment and report latency percentiles
    and throughput. Designed to compare the old env vs the new fastrag env on prod.

setup mode
    Create a benchmark custom model, deploy it twice (once on FastDRUM/FRAG, once on DRUM),
    and save the deployment IDs for use with `compare`. Idempotent: pass --model-id to
    reuse an existing model and skip re-creation.

    --rag flag: RAG-representative benchmark (50ms VDB + 200ms LLM, ~250ms total).
    Both deployments use the same python312_genai env; FRAG is enabled via
    DR_GENAI_RAG_FRAG_RUNNER=1 injected on the FRAG deployment. Pass the
    python312_genai env ID via --fastrag-env-id (run list-envs to find it).

compare mode
    Run the load test on both the FastDRUM and DRUM deployments from `setup` at multiple
    concurrency levels and print a side-by-side comparison table. Reads deployment IDs
    from ~/.config/fastrag/loadtest_deployments.json (written by setup) or accepts
    --fastrag-id / --drum-id directly.

Usage:
    # Test one classic-env model (default)
    uv run scripts/test_models.py correctness

    # Test a specific model
    uv run scripts/test_models.py correctness --model-id <ID> --dataset-id <ID>

    # Test all classic-env models
    uv run scripts/test_models.py correctness --limit 0

    # Load test a deployment
    uv run scripts/test_models.py load --deployment-id <ID> --requests 100 --concurrency 20

    # Create benchmark deployments (FastDRUM + DRUM)
    uv run scripts/test_models.py setup

    # Create RAG benchmark deployments (FRAG vs DRUM, same python312_genai env)
    uv run scripts/test_models.py setup --rag --fastrag-env-id <python312_genai_env_id>

    # Compare FastDRUM/FRAG vs DRUM across concurrency levels
    uv run scripts/test_models.py compare

Credentials: env vars DATAROBOT_ENDPOINT / DATAROBOT_API_TOKEN,
             or .env file via --env-file,
             or ~/.config/datarobot/drconfig.yaml (from `datarobot auth login`).
"""

import argparse
import asyncio
import io
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from statistics import mean, quantiles

import aiohttp
import requests

REPO_ROOT = Path(__file__).parent.parent
PERSONAL_ENV_CONFIG = Path.home() / ".config" / "fastrag" / "personal_env.json"
LOADTEST_CONFIG = Path.home() / ".config" / "fastrag" / "loadtest_deployments.json"
DR_CONFIG = Path.home() / ".config" / "datarobot" / "drconfig.yaml"
BENCHMARK_MODEL_DIR = REPO_ROOT / "tests" / "benchmark_model"
BENCHMARK_MODEL_RAG_DIR = REPO_ROOT / "tests" / "benchmark_model_rag"
BENCHMARK_MODEL_RAG_SYNC_DIR = REPO_ROOT / "tests" / "benchmark_model_rag_sync"

# Classic DataRobot environments (not fastrag) — baseline candidates
CLASSIC_ENV_IDS = {
    "680fe4949604e9eba46b1775",  # [DataRobot] Python 3.11 GenAI Agents
}
DRUM_ENV_ID = "680fe4949604e9eba46b1775"  # [DataRobot] Python 3.11 GenAI Agents

POLL_INTERVAL = 10
POLL_TIMEOUT = 600
DEPLOY_TIMEOUT = 1800  # Serverless deployments can take 20+ min


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def load_credentials(env_file: str | None = None) -> tuple[str, str]:
    if env_file:
        for line in Path(env_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    endpoint = os.environ.get("DATAROBOT_ENDPOINT", "")
    token = os.environ.get("DATAROBOT_API_TOKEN", "")

    if not (endpoint and token) and DR_CONFIG.exists():
        import yaml
        cfg = yaml.safe_load(DR_CONFIG.read_text()) or {}
        endpoint = endpoint or cfg.get("endpoint", "")
        token = token or cfg.get("token", "")

    if not endpoint or not token:
        print("ERROR: credentials not found. Set DATAROBOT_ENDPOINT / DATAROBOT_API_TOKEN "
              "or run `datarobot auth login`.", file=sys.stderr)
        sys.exit(1)

    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/api/v2"):
        endpoint = endpoint[: -len("/api/v2")]
    return endpoint, token


def load_personal_env(endpoint: str) -> tuple[str, str]:
    if not PERSONAL_ENV_CONFIG.exists():
        print("ERROR: no personal env found. Run `make upload` first.", file=sys.stderr)
        sys.exit(1)
    cfg = json.loads(PERSONAL_ENV_CONFIG.read_text())
    if cfg.get("endpoint") != endpoint.rstrip("/"):
        print(f"WARNING: personal env was created on {cfg.get('endpoint')}, "
              f"not {endpoint}. Use --fastrag-env-id to override.")
    return cfg["id"], cfg["name"]


# ---------------------------------------------------------------------------
# DataRobot API client
# ---------------------------------------------------------------------------


class DR:
    def __init__(self, endpoint: str, token: str):
        self.endpoint = endpoint
        self.token = token
        self._s = requests.Session()
        self._s.headers["Authorization"] = f"Bearer {token}"
        self._read_timeout = 60  # seconds per HTTP request

    def _url(self, path: str) -> str:
        return f"{self.endpoint}/api/v2{path}"

    def get(self, path: str, **kw) -> dict:
        r = self._s.get(self._url(path), timeout=self._read_timeout, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, **kw) -> dict:
        r = self._s.post(self._url(path), **kw)
        if not r.ok:
            print(f"ERROR POST {path}: {r.status_code} {r.text[:300]}", file=sys.stderr)
            sys.exit(1)
        return r.json()

    def list_models(self, limit: int = 100) -> list:
        return self.get(f"/customModels/?limit={limit}")["data"]

    def download_version_files(self, model_id: str, version_id: str, dest: str) -> None:
        r = self._s.get(self._url(f"/customModels/{model_id}/versions/{version_id}/download/"))
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(dest)

    def create_version(self, model_id: str, files_dir: str, env_id: str) -> dict:
        form: list = [
            ("baseEnvironmentId", (None, env_id)),
            ("isMajorUpdate", (None, "false")),
        ]
        handles = []
        for f in sorted(Path(files_dir).rglob("*")):
            if f.is_file():
                rel = str(f.relative_to(files_dir))
                form.append(("filePath", (None, rel)))
                h = open(f, "rb")
                handles.append(h)
                form.append(("file", (f.name, h, "application/octet-stream")))
        try:
            r = self._s.post(self._url(f"/customModels/{model_id}/versions/"), files=form)
            if not r.ok:
                print(f"ERROR create version: {r.status_code} {r.text[:300]}", file=sys.stderr)
                sys.exit(1)
            return r.json()
        finally:
            for h in handles:
                h.close()

    def trigger_test(self, model_id: str, version_id: str, dataset_id: str | None = None) -> None:
        payload: dict = {"customModelId": model_id, "customModelVersionId": version_id}
        if dataset_id:
            payload["datasetId"] = dataset_id
        self.post("/customModelTests/", json=payload)

    def poll_test(self, model_id: str, version_id: str, timeout: int = POLL_TIMEOUT) -> dict | None:
        """Poll until the test for this specific version completes."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.get(f"/customModelTests/?customModelId={model_id}")["data"]
            for t in data:
                # customModelImageId is the version ID in the test record
                if t.get("customModelImageId") == version_id:
                    if t["overallStatus"] in ("succeeded", "failed"):
                        return t
                    break  # found it, still in progress
            time.sleep(POLL_INTERVAL)
        return None

    def upload_dataset(self, content: str, name: str = "test_data.csv") -> str:
        r = self._s.post(
            self._url("/datasets/fromFile/"),
            files={"file": (name, content.encode(), "text/csv")},
        )
        r.raise_for_status()
        return r.json()["catalogId"]

    def create_model(self, name: str, target_type: str = "TextGeneration") -> dict:
        return self.post("/customModels/", json={
            "name": name,
            "targetType": target_type,
            "customModelType": "inference",
            "targetName": "response",
        })

    def poll_version_build(self, model_id: str, version_id: str, timeout: int = POLL_TIMEOUT) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            v = self.get(f"/customModels/{model_id}/versions/{version_id}/")
            status = v.get("buildStatus")
            # Custom model versions have no build step — None means immediately ready.
            # Execution environment versions use "success"/"failed".
            if status is None or status == "success":
                return True
            if status == "failed":
                print(f"\n  Build log: {v.get('buildLog', '')[:400]}", file=sys.stderr)
                return False
            print(".", end="", flush=True)
            time.sleep(POLL_INTERVAL)
        return False

    def package_version(self, version_id: str, name: str) -> dict:
        return self.post("/modelPackages/fromCustomModelVersion/", json={
            "customModelVersionId": version_id,
            "name": name,
        })

    def poll_model_package_build(self, pkg_id: str, timeout: int = POLL_TIMEOUT) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            pkg = self.get(f"/modelPackages/{pkg_id}/")
            status = pkg.get("buildStatus")
            if status == "complete":
                return True
            if status == "failed":
                print(f"\n  Model package build failed", file=sys.stderr)
                return False
            print(".", end="", flush=True)
            time.sleep(POLL_INTERVAL)
        return False

    def list_prediction_environments(self) -> list:
        return self.get("/predictionEnvironments/?limit=100")["data"]

    def create_deployment(
        self, model_package_id: str, pred_env_id: str, label: str, timeout: int = POLL_TIMEOUT
    ) -> str:
        """Create a deployment from a model package. Returns deployment ID."""
        r = self._s.post(self._url("/deployments/fromModelPackage/"), json={
            "modelPackageId": model_package_id,
            "predictionEnvironmentId": pred_env_id,
            "label": label,
            "importance": "LOW",
        }, timeout=self._read_timeout)
        if not r.ok:
            print(f"ERROR POST /deployments/fromModelPackage/: {r.status_code} {r.text[:300]}",
                  file=sys.stderr)
            sys.exit(1)
        # 201 returns body with id; 202 is async — poll the Location header for the job
        if r.status_code == 202:
            job_url = r.headers.get("Location", "")
            if not job_url:
                print("ERROR: 202 response but no Location header", file=sys.stderr)
                sys.exit(1)
            return self._poll_async_job(job_url, timeout=timeout)
        return r.json()["id"]

    def _poll_async_job(self, job_url: str, timeout: int = POLL_TIMEOUT) -> str:
        """Poll an async job URL until complete. Returns the created resource ID."""
        deadline = time.time() + timeout
        last_data = {}
        n = 0
        while time.time() < deadline:
            r = self._s.get(job_url, timeout=self._read_timeout)
            if not r.ok:
                print(f"ERROR polling job {job_url}: {r.status_code} {r.text[:200]}", file=sys.stderr)
                sys.exit(1)
            last_data = r.json()
            status = last_data.get("status", "")
            if n == 0:
                print(f"[job status={status}]", end="", flush=True)
            # DR's deployments/fromModelPackage job returns the deployment resource
            # directly (status="active") rather than a generic job status object.
            if status in ("COMPLETED", "SUCCEEDED", "succeeded", "completed", "active", "inactive"):
                dep_id = (
                    last_data.get("id")
                    or last_data.get("entityId")
                    or (last_data.get("data") or {}).get("id")
                    or last_data.get("status_id")
                )
                if not dep_id:
                    print(f"\nERROR: job COMPLETED but no id found in: {last_data}", file=sys.stderr)
                    sys.exit(1)
                return dep_id
            if status in ("ERROR", "FAILED", "failed", "error", "ABORTED", "EXPIRED"):
                print(f"\nAsync job failed: {last_data}", file=sys.stderr)
                sys.exit(1)
            print(".", end="", flush=True)
            time.sleep(POLL_INTERVAL)
            n += 1
        print(f" timed out. Last response: {last_data}", file=sys.stderr)
        sys.exit(1)

    def get_deployment_prediction_key(self, deployment_id: str) -> str | None:
        """Try to extract the prediction server key from the deployment object."""
        d = self.get(f"/deployments/{deployment_id}/")
        # Try common field paths where DR stores the prediction key
        for path in [
            ["defaultPredictionServer", "datarobot-key"],
            ["defaultPredictionServer", "serverKey"],
            ["predictionEnvironment", "datarobotKey"],
            ["predictionEnvironment", "serverKey"],
        ]:
            node = d
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            if node and isinstance(node, str):
                return node
        return None

    def list_execution_environments(self) -> list:
        return self.get("/executionEnvironments/?limit=100")["data"]

    def list_deployments(self, search: str = "") -> list:
        params = "limit=100"
        if search:
            params += f"&search={search}"
        return self.get(f"/deployments/?{params}").get("data", [])

    def delete_deployment(self, dep_id: str) -> None:
        r = self._s.delete(self._url(f"/deployments/{dep_id}/"))
        if not r.ok and r.status_code != 404:
            print(f"  WARN: DELETE /deployments/{dep_id}/: {r.status_code} {r.text[:200]}", file=sys.stderr)

    def poll_deployment_active(self, deployment_id: str, timeout: int = POLL_TIMEOUT) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            d = self.get(f"/deployments/{deployment_id}/")
            if d.get("status") == "active":
                return True
            print(".", end="", flush=True)
            time.sleep(POLL_INTERVAL)
        return False

    def set_deployment_env_vars(self, deployment_id: str, vars_dict: dict) -> bool:
        """Set environment variables on a deployment. Returns True on success."""
        values = [{"name": k, "value": v} for k, v in vars_dict.items()]
        r = self._s.put(
            self._url(f"/deployments/{deployment_id}/deploymentEnvironmentVariableValues/"),
            json={"values": values},
            timeout=self._read_timeout,
        )
        if not r.ok:
            print(f"\n  WARN: could not set env vars (HTTP {r.status_code}): {r.text[:200]}",
                  file=sys.stderr)
            return False
        return True


# ---------------------------------------------------------------------------
# Correctness testing
# ---------------------------------------------------------------------------


def _fmt_checks(testing_status: dict) -> str:
    return "  ".join(
        f"{k}:{v['status'][0].upper()}"
        for k, v in testing_status.items()
        if v["status"] != "skipped"
    )


def _run_and_time_test(
    dr: DR, model_id: str, version_id: str, dataset_id: str | None, label: str
) -> tuple[str, float, str]:
    """Trigger a test, poll to completion. Returns (status, elapsed_s, checks_summary)."""
    t0 = time.time()
    dr.trigger_test(model_id, version_id, dataset_id)
    result = dr.poll_test(model_id, version_id)
    elapsed = time.time() - t0
    if result is None:
        return "timeout", elapsed, ""
    return result["overallStatus"], elapsed, _fmt_checks(result["testingStatus"])


def test_one_model(
    dr: DR,
    model: dict,
    fastrag_env_id: str,
    dataset_id: str | None,
) -> dict:
    model_id = model["id"]
    name = model["name"]
    latest = model["latestVersion"]
    current_vid = latest["id"]

    print(f"\n{'─'*60}")
    print(f"  {name}  ({model['targetType']})")
    print(f"  Current env: {latest['baseEnvironmentId']}  version: {latest['label']}")

    # Step 1: baseline
    print(f"  [1/3] Baseline...", end=" ", flush=True)
    base_status, base_t, base_checks = _run_and_time_test(
        dr, model_id, current_vid, dataset_id, "baseline"
    )
    icon = "✓" if base_status == "succeeded" else "✗"
    print(f"{icon} {base_status} in {base_t:.1f}s  [{base_checks}]")

    if base_status != "succeeded":
        print("  Baseline failed — skipping fastrag test (fix the model or provide a dataset)")
        return {
            "model": name, "target_type": model["targetType"],
            "baseline": base_status, "baseline_s": base_t,
            "fastrag": "skipped", "fastrag_s": None,
        }

    # Step 2: create version with fastrag env
    print(f"  [2/3] Creating fastrag version...", end=" ", flush=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        dr.download_version_files(model_id, current_vid, tmpdir)
        new_v = dr.create_version(model_id, tmpdir, fastrag_env_id)
    new_vid = new_v["id"]
    print(f"created {new_v['label']} in {time.time()-t0:.1f}s")

    # Step 3: fastrag test
    print(f"  [3/3] Fastdrum test...", end=" ", flush=True)
    fd_status, fd_t, fd_checks = _run_and_time_test(
        dr, model_id, new_vid, dataset_id, "fastrag"
    )
    icon = "✓" if fd_status == "succeeded" else "✗"
    print(f"{icon} {fd_status} in {fd_t:.1f}s  [{fd_checks}]")

    return {
        "model": name, "target_type": model["targetType"],
        "baseline": base_status, "baseline_s": base_t,
        "fastrag": fd_status, "fastrag_s": fd_t,
    }


def cmd_correctness(args, dr: DR, fastrag_env_id: str) -> None:
    models = dr.list_models()

    if args.model_id:
        models = [m for m in models if m["id"] == args.model_id]
    elif args.env_id:
        models = [m for m in models if m["latestVersion"]["baseEnvironmentId"] == args.env_id]
    else:
        models = [m for m in models if m["latestVersion"]["baseEnvironmentId"] in CLASSIC_ENV_IDS]

    limit = args.limit
    if limit and limit > 0:
        models = models[:limit]

    if not models:
        print("No models found. Use --model-id or --env-id to specify, "
              "or add the target env to CLASSIC_ENV_IDS.")
        return

    # Dataset
    dataset_id = args.dataset_id
    if not dataset_id and args.dataset_file:
        print(f"Uploading {args.dataset_file}...")
        dataset_id = dr.upload_dataset(Path(args.dataset_file).read_text(), Path(args.dataset_file).name)
        print(f"  Dataset ID: {dataset_id}")

    print(f"\nTesting {len(models)} model(s)")
    print(f"Fastdrum env: {fastrag_env_id}")
    if dataset_id:
        print(f"Dataset:      {dataset_id}")

    results = [test_one_model(dr, m, fastrag_env_id, dataset_id) for m in models]

    # Summary
    print(f"\n{'═'*80}")
    print("SUMMARY")
    print(f"{'Model':<40} {'Type':<18} {'Baseline':>10} {'t':>7}  {'Fastdrum':>10} {'t':>7}")
    print("─" * 80)
    for r in results:
        bt = f"{r['baseline_s']:.1f}s" if r.get("baseline_s") is not None else "-"
        ft = f"{r['fastrag_s']:.1f}s" if r.get("fastrag_s") is not None else "-"
        print(
            f"{r['model']:<40} {r['target_type']:<18} "
            f"{r['baseline']:>10} {bt:>7}  {r['fastrag']:>10} {ft:>7}"
        )


# ---------------------------------------------------------------------------
# Load testing
# ---------------------------------------------------------------------------


async def _load_worker(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    payload: dict,
    sem: asyncio.Semaphore,
    results: list,
) -> None:
    async with sem:
        t0 = time.time()
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                body = await resp.read()
                elapsed = time.time() - t0
                entry: dict = {"status": resp.status, "elapsed": elapsed, "ok": resp.status < 400}
                if resp.status >= 400:
                    entry["error_body"] = body[:400].decode("utf-8", errors="replace")
                results.append(entry)
        except Exception as exc:
            results.append({"status": 0, "elapsed": time.time() - t0, "ok": False, "error": str(exc)})


async def _async_load(
    endpoint: str,
    token: str,
    deployment_id: str,
    payload: dict,
    n: int,
    concurrency: int,
    datarobot_key: str | None = None,
) -> tuple[list, float]:
    url = f"{endpoint}/api/v2/deployments/{deployment_id}/chat/completions/"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if datarobot_key:
        headers["datarobot-key"] = datarobot_key
    results: list = []
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.time()
        await asyncio.gather(*[
            _load_worker(session, url, headers, payload, sem, results)
            for _ in range(n)
        ])
        return results, time.time() - t0


def cmd_load(args, endpoint: str, token: str) -> None:
    payload: dict
    if args.payload:
        payload = json.loads(Path(args.payload).read_text())
    else:
        payload = {"model": "datarobot-deployed-llm", "messages": [{"role": "user", "content": "hello"}]}

    n = args.requests
    c = args.concurrency
    print(f"\nLoad test: deployment {args.deployment_id}")
    print(f"  {n} requests  concurrency {c}  payload: {json.dumps(payload)[:80]}")

    results, total = asyncio.run(
        _async_load(endpoint, token, args.deployment_id, payload, n, c,
                    datarobot_key=getattr(args, "datarobot_key", None))
    )

    latencies = sorted(r["elapsed"] for r in results)
    errors = sum(1 for r in results if not r["ok"])
    qs = quantiles(latencies, n=100) if len(latencies) > 1 else [latencies[0]] * 99

    error_samples = [r for r in results if not r["ok"]]
    if error_samples:
        sample = error_samples[0]
        print(f"\n  First error: HTTP {sample.get('status')} — {sample.get('error_body') or sample.get('error', '')}")

    print(f"\n  Requests:    {n}  ({errors} errors, {errors/n*100:.1f}%)")
    print(f"  Total time:  {total:.2f}s")
    print(f"  Throughput:  {n/total:.1f} req/s")
    print(f"  Latency:")
    print(f"    min  {min(latencies)*1000:>7.0f} ms")
    print(f"    p50  {qs[49]*1000:>7.0f} ms")
    print(f"    p95  {qs[94]*1000:>7.0f} ms")
    print(f"    p99  {qs[98]*1000:>7.0f} ms")
    print(f"    max  {max(latencies)*1000:>7.0f} ms")
    print(f"    mean {mean(latencies)*1000:>7.0f} ms")


# ---------------------------------------------------------------------------
# Setup: create benchmark deployments
# ---------------------------------------------------------------------------


def _deploy_version(
    dr: DR, model_id: str, code_dir: str, env_id: str, label: str, pred_env_id: str
) -> str:
    """Create a version, build it, package it, deploy it. Returns deployment ID."""
    print(f"  Creating {label} version (env {env_id})...", end=" ", flush=True)
    version = dr.create_version(model_id, code_dir, env_id)
    vid = version["id"]
    print(f"v{version['label']}")

    print(f"  Building", end="", flush=True)
    if not dr.poll_version_build(model_id, vid):
        print(f"\nERROR: {label} version build failed.", file=sys.stderr)
        sys.exit(1)
    print(" done")

    print(f"  Packaging {label}...", end=" ", flush=True)
    pkg = dr.package_version(vid, f"[Load Test] {label}")
    print(f"package {pkg['id']}")

    print(f"  Building model package", end="", flush=True)
    if not dr.poll_model_package_build(pkg["id"]):
        print(f"\nERROR: {label} model package build failed.", file=sys.stderr)
        sys.exit(1)
    print(" done")

    print(f"  Deploying {label}...", end=" ", flush=True)
    dep_id = dr.create_deployment(pkg["id"], pred_env_id, f"[FastDRUM Load Test] {label}", timeout=DEPLOY_TIMEOUT)
    print(f"deployment {dep_id}")
    print(f"  Waiting for {label} to go active", end="", flush=True)
    if not dr.poll_deployment_active(dep_id, timeout=DEPLOY_TIMEOUT):
        print(f"\nERROR: {label} deployment did not become active (id={dep_id}).", file=sys.stderr)
        sys.exit(1)
    print(f" done")
    return dep_id


def cmd_setup(args, dr: DR, fastrag_env_id: str, endpoint: str) -> None:
    rag_mode = getattr(args, "rag", False)

    # 1. Create or reuse model
    if args.model_id:
        model = dr.get(f"/customModels/{args.model_id}/")
        print(f"Reusing model: {model['name']} ({args.model_id})")
        model_id = args.model_id
    else:
        label = "[FastDRUM Load Test] RAG Benchmark" if rag_mode else "[FastDRUM Load Test] Benchmark Chat"
        print(f"Creating benchmark custom model: {label}")
        model = dr.create_model(label)
        model_id = model["id"]
        print(f"  Created: {model_id}")

    if rag_mode:
        # RAG mode: both deployments use the same python312_genai env.
        # FRAG is enabled by injecting DR_GENAI_RAG_FRAG_RUNNER=1 after deployment.
        # Pass --fastrag-env-id with the python312_genai env ID from DataRobot.
        code_dir = args.code_dir or str(BENCHMARK_MODEL_RAG_DIR)
        drum_code_dir = args.drum_code_dir or str(BENCHMARK_MODEL_RAG_SYNC_DIR)
        drum_env_id = args.drum_env_id or fastrag_env_id  # same env, env var absent → DRUM
    else:
        code_dir = args.code_dir or str(BENCHMARK_MODEL_DIR)
        drum_code_dir = args.drum_code_dir or str(BENCHMARK_MODEL_DIR.parent / "benchmark_model_sync")
        drum_env_id = args.drum_env_id or DRUM_ENV_ID

    # 2. Find prediction environment
    pred_env_id = args.pred_env_id
    envs = dr.list_prediction_environments()
    available = [e for e in envs if not e.get("isDeleted")]
    if not pred_env_id:
        if not available:
            print("ERROR: no prediction environments found. Pass --pred-env-id.", file=sys.stderr)
            sys.exit(1)
        print("Available prediction environments:")
        for e in available:
            print(f"  {e['id']}  {e.get('name', '(no name)')}  platform={e.get('platform', '?')}")
        pred_env = available[0]
        pred_env_id = pred_env["id"]
        print(f"Using: {pred_env.get('name', pred_env_id)} ({pred_env_id})")
        print("  (override with --pred-env-id <id>)")
    else:
        pred_env = next((e for e in available if e["id"] == pred_env_id), {"id": pred_env_id})
        print(f"Using prediction environment: {pred_env.get('name', pred_env_id)} ({pred_env_id})")

    # 3. Clean up any stale load-test deployments so they don't block the queue
    keep_ids = {args.fastrag_dep_id, args.drum_dep_id} - {None}
    stale = [d for d in dr.list_deployments(search="[FastDRUM Load Test]")
             if d["id"] not in keep_ids]
    if stale:
        print(f"Cleaning up {len(stale)} stale load-test deployment(s)...")
        for d in stale:
            print(f"  Deleting {d['id']} ({d.get('label', '')})")
            dr.delete_deployment(d["id"])

    # 4. Deploy FastDRUM + DRUM (skip if existing dep ID supplied)
    print()
    if args.fastrag_dep_id:
        fd_dep_id = args.fastrag_dep_id
        print(f"Reusing FastDRUM deployment: {fd_dep_id}")
    else:
        fd_dep_id = _deploy_version(dr, model_id, code_dir, fastrag_env_id, "FRAG" if rag_mode else "FastDRUM", pred_env_id)

    if rag_mode and not args.fastrag_dep_id:
        print(f"  Injecting DR_GENAI_RAG_FRAG_RUNNER=1...", end=" ", flush=True)
        ok = dr.set_deployment_env_vars(fd_dep_id, {"DR_GENAI_RAG_FRAG_RUNNER": "1"})
        if ok:
            print("done")
        else:
            print("skipped (env var API not available; if your env always runs fastrag this is fine)")

    print()
    if args.drum_dep_id:
        drum_dep_id = args.drum_dep_id
        print(f"Reusing DRUM deployment: {drum_dep_id}")
    else:
        drum_dep_id = _deploy_version(dr, model_id, drum_code_dir, drum_env_id, "DRUM", pred_env_id)

    # 5. Save and summarise
    config = {
        "endpoint": endpoint,
        "model_id": model_id,
        "fastrag": {"deployment_id": fd_dep_id, "env_id": fastrag_env_id},
        "drum": {"deployment_id": drum_dep_id, "env_id": drum_env_id},
    }
    LOADTEST_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    LOADTEST_CONFIG.write_text(json.dumps(config, indent=2))

    print(f"\n{'═'*60}")
    print("Setup complete.")
    print(f"  FastDRUM: {endpoint}/api/v2/deployments/{fd_dep_id}/chat/completions/")
    print(f"  DRUM:     {endpoint}/api/v2/deployments/{drum_dep_id}/chat/completions/")
    print(f"\nRun the comparison:")
    print(f"  uv run scripts/test_models.py compare")
    print(f"Config saved: {LOADTEST_CONFIG}")


# ---------------------------------------------------------------------------
# Compare: side-by-side load test across concurrency levels
# ---------------------------------------------------------------------------


def _summarize(results: list, total: float, n: int) -> dict:
    latencies = sorted(r["elapsed"] for r in results)
    errors = sum(1 for r in results if not r["ok"])
    qs = quantiles(latencies, n=100) if len(latencies) > 1 else [latencies[0]] * 99
    return {
        "rps": n / total,
        "p50_ms": qs[49] * 1000,
        "p95_ms": qs[94] * 1000,
        "p99_ms": qs[98] * 1000,
        "err_pct": errors / n * 100,
    }


def cmd_compare(args, endpoint: str, token: str) -> None:  # noqa: DR creates DR client internally
    # Resolve deployment IDs
    if args.fastrag_id and args.drum_id:
        fd_dep_id = args.fastrag_id
        drum_dep_id = args.drum_id
    elif LOADTEST_CONFIG.exists():
        cfg = json.loads(LOADTEST_CONFIG.read_text())
        if cfg.get("endpoint") != endpoint:
            print(f"WARNING: config was created for {cfg.get('endpoint')}, not {endpoint}.")
        fd_dep_id = cfg["fastrag"]["deployment_id"]
        drum_dep_id = cfg["drum"]["deployment_id"]
    else:
        print("ERROR: no deployment IDs. Run `setup` first or pass --fastrag-id and --drum-id.",
              file=sys.stderr)
        sys.exit(1)

    dr = DR(endpoint, token)
    datarobot_key = getattr(args, "datarobot_key", None)
    if not datarobot_key:
        datarobot_key = dr.get_deployment_prediction_key(fd_dep_id)
        if datarobot_key:
            print(f"  [auto] datarobot-key: {datarobot_key[:8]}…")
        else:
            print("  WARN: could not auto-fetch datarobot-key; if you get 403, pass --datarobot-key")

    payload = json.loads(Path(args.payload).read_text()) if args.payload else \
        {"model": "datarobot-deployed-llm", "messages": [{"role": "user", "content": "hello"}]}
    concurrency_levels = [int(c) for c in args.concurrency.split(",")]
    n = args.requests

    print(f"\nFastDRUM: {fd_dep_id}")
    print(f"DRUM:     {drum_dep_id}")
    print(f"Requests: {n} per run  |  Concurrency levels: {concurrency_levels}")
    print(f"Payload:  {json.dumps(payload)[:80]}\n")

    rows = []
    for c in concurrency_levels:
        print(f"  c={c:<3}", end="  ", flush=True)

        print("FastDRUM", end="", flush=True)
        fd_results, fd_total = asyncio.run(_async_load(endpoint, token, fd_dep_id, payload, n, c, datarobot_key=datarobot_key))
        fd = _summarize(fd_results, fd_total, n)
        print(f" {fd['rps']:5.1f} rps  p95={fd['p95_ms']:5.0f}ms  err={fd['err_pct']:4.1f}%", end="")
        fd_err = next((r for r in fd_results if not r["ok"]), None)

        print("  |  DRUM", end="", flush=True)
        drum_results, drum_total = asyncio.run(_async_load(endpoint, token, drum_dep_id, payload, n, c, datarobot_key=datarobot_key))
        drum = _summarize(drum_results, drum_total, n)
        print(f" {drum['rps']:5.1f} rps  p95={drum['p95_ms']:5.0f}ms  err={drum['err_pct']:4.1f}%")
        drum_err = next((r for r in drum_results if not r["ok"]), None)

        if c == concurrency_levels[0]:
            if fd_err:
                print(f"  [FastDRUM error sample] HTTP {fd_err.get('status')} — {fd_err.get('error_body') or fd_err.get('error', '')}")
            if drum_err:
                print(f"  [DRUM error sample]     HTTP {drum_err.get('status')} — {drum_err.get('error_body') or drum_err.get('error', '')}")

        rows.append({"c": c, "fd": fd, "drum": drum})

    # Summary table
    print(f"\n{'═'*88}")
    print(f"{'Concurrency':>11}  {'FD rps':>8}  {'DR rps':>8}  {'Speedup':>8}  "
          f"{'FD p95':>8}  {'DR p95':>8}  {'FD err%':>8}  {'DR err%':>8}")
    print("─" * 88)
    for r in rows:
        fd, drum = r["fd"], r["drum"]
        speedup = fd["rps"] / drum["rps"] if drum["rps"] > 0 else float("inf")
        print(f"{r['c']:>11}  {fd['rps']:>8.1f}  {drum['rps']:>8.1f}  {speedup:>7.2f}×  "
              f"{fd['p95_ms']:>8.0f}  {drum['p95_ms']:>8.0f}  {fd['err_pct']:>8.1f}  {drum['err_pct']:>8.1f}")

    # Verdict
    avg_speedup = sum(r["fd"]["rps"] / r["drum"]["rps"] for r in rows if r["drum"]["rps"] > 0) / len(rows)
    print(f"\n  Average speedup: {avg_speedup:.2f}×  ", end="")
    if avg_speedup >= 2.0:
        print("✓ meets the 2× target")
    else:
        print(f"✗ below the 2× target (need {2/avg_speedup:.1f}× more headroom)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env-file", help="Load credentials from a .env file (e.g. .env.staging)")

    sub = parser.add_subparsers(dest="command", required=True)

    # correctness
    cp = sub.add_parser("correctness", help="Baseline + fastrag correctness test with timing")
    cp.add_argument("--model-id", help="Test a specific model by ID")
    cp.add_argument("--env-id", help="Test all models using this execution environment ID")
    cp.add_argument("--limit", type=int, default=1,
                    help="Max models to test; 0 = all found (default: 1)")
    cp.add_argument("--dataset-id", help="Reuse an already-uploaded DataRobot dataset ID")
    cp.add_argument("--dataset-file", help="Local CSV to upload and use as test data")
    cp.add_argument("--fastrag-env-id", help="Override the fastrag env ID from personal_env.json")

    # load
    lp = sub.add_parser("load", help="Load test a deployment endpoint")
    lp.add_argument("--deployment-id", required=True, help="DataRobot deployment ID")
    lp.add_argument("--requests", type=int, default=50, help="Total requests (default: 50)")
    lp.add_argument("--concurrency", type=int, default=10, help="Concurrent requests (default: 10)")
    lp.add_argument("--payload", help="JSON file with the request body")
    lp.add_argument("--datarobot-key", help="datarobot-key header for predApi auth")

    # setup
    sp = sub.add_parser("setup", help="Create FastDRUM + DRUM benchmark deployments")
    sp.add_argument("--model-id", help="Reuse an existing custom model (skip creation)")
    sp.add_argument("--rag", action="store_true",
                    help="RAG mode: use async/sync RAG benchmark models and the python312_genai env. "
                         "Pass --fastrag-env-id with the python312_genai env ID from DataRobot. "
                         "Both deployments use the same env; FRAG is toggled via DR_GENAI_RAG_FRAG_RUNNER=1 "
                         "injected on the FRAG deployment after creation.")
    sp.add_argument("--code-dir", help="Directory with custom.py for FRAG/FastDRUM (default depends on --rag)")
    sp.add_argument("--drum-code-dir", help="Directory with custom.py for DRUM (default depends on --rag)")
    sp.add_argument("--drum-env-id", help="DRUM environment ID (default: same as --fastrag-env-id in RAG mode, "
                                          f"otherwise {DRUM_ENV_ID})")
    sp.add_argument("--fastrag-env-id", help="FRAG/FastDRUM environment ID (default: personal env). "
                                               "In RAG mode, pass the python312_genai env ID from DataRobot "
                                               "(run list-envs to find it after the env is built from the PR).")
    sp.add_argument("--pred-env-id", help="Prediction environment ID (default: first available)")
    sp.add_argument("--fastrag-dep-id", help="Reuse an existing FRAG/FastDRUM deployment (skip creation)")
    sp.add_argument("--drum-dep-id", help="Reuse an existing DRUM deployment (skip creation)")

    # list-envs
    sub.add_parser("list-envs", help="List available execution environments (base envs for custom models)")

    # show-deployment
    sdp = sub.add_parser("show-deployment", help="Dump deployment JSON to find prediction key fields")
    sdp.add_argument("deployment_id", help="Deployment ID to inspect")

    # compare
    cp2 = sub.add_parser("compare", help="Side-by-side load test: FastDRUM vs DRUM")
    cp2.add_argument("--fastrag-id", help="FastDRUM deployment ID (overrides saved config)")
    cp2.add_argument("--drum-id", help="DRUM deployment ID (overrides saved config)")
    cp2.add_argument("--requests", type=int, default=100, help="Requests per run (default: 100)")
    cp2.add_argument("--concurrency", default="10,20,50",
                     help="Comma-separated concurrency levels (default: 10,20,50)")
    cp2.add_argument("--payload", help="JSON file with the request body")
    cp2.add_argument("--datarobot-key", help="datarobot-key header for predApi auth (auto-fetched if omitted)")

    args = parser.parse_args()
    endpoint, token = load_credentials(args.env_file)

    if args.command == "correctness":
        fastrag_env_id = args.fastrag_env_id or load_personal_env(endpoint)[0]
        cmd_correctness(args, DR(endpoint, token), fastrag_env_id)
    elif args.command == "load":
        cmd_load(args, endpoint, token)
    elif args.command == "setup":
        fastrag_env_id = args.fastrag_env_id or load_personal_env(endpoint)[0]
        cmd_setup(args, DR(endpoint, token), fastrag_env_id, endpoint)
    elif args.command == "show-deployment":
        dr = DR(endpoint, token)
        d = dr.get(f"/deployments/{args.deployment_id}/")
        print(json.dumps(d, indent=2))
    elif args.command == "list-envs":
        dr = DR(endpoint, token)
        envs = dr.list_execution_environments()
        print(f"{'ID':<26}  {'Name'}")
        print("─" * 80)
        for e in envs:
            print(f"{e['id']:<26}  {e.get('name', '(no name)')}")
    elif args.command == "compare":
        cmd_compare(args, endpoint, token)


if __name__ == "__main__":
    main()
