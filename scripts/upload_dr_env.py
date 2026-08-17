#!/usr/bin/env python3
"""
Upload a new execution environment version to DataRobot.

Default: uploads to a private personal environment (created on first run).
         Personal env ID is stored in ~/.config/fastrag/personal_env.json
         so it persists across runs without being committed to git.

--promote: uploads to the shared public environment in env_info.json.
           Use this only when the change is ready for everyone.

Usage:
    uv run scripts/upload_dr_env.py             # personal env (safe to iterate)
    uv run scripts/upload_dr_env.py --promote   # public env (affects all users)

Credentials are read from env vars if set, otherwise from ~/.config/datarobot/drconfig.yaml
(written by `datarobot auth login`):
    DATAROBOT_API_TOKEN   — your DataRobot API token
    DATAROBOT_ENDPOINT    — e.g. https://app.datarobot.com  (with or without /api/v2)
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).parent.parent
ENV_DIR = REPO_ROOT / "examples" / "fastrag_py323_genai_env"
ENV_INFO = ENV_DIR / "env_info.json"
PERSONAL_ENV_CONFIG = Path.home() / ".config" / "fastrag" / "personal_env.json"

POLL_INTERVAL = 10
POLL_TIMEOUT = 900


_DR_CONFIG = Path.home() / ".config" / "datarobot" / "drconfig.yaml"


def _load_dr_credentials() -> tuple[str, str]:
    """Return (endpoint, token) from env vars, falling back to drconfig.yaml."""
    endpoint = os.environ.get("DATAROBOT_ENDPOINT")
    token = os.environ.get("DATAROBOT_API_TOKEN")

    if not (endpoint and token) and _DR_CONFIG.exists():
        import yaml
        cfg = yaml.safe_load(_DR_CONFIG.read_text()) or {}
        endpoint = endpoint or cfg.get("endpoint", "")
        token = token or cfg.get("token", "")

    if not endpoint:
        print("ERROR: no endpoint found. Set DATAROBOT_ENDPOINT or run `datarobot auth login`.",
              file=sys.stderr)
        sys.exit(1)
    if not token:
        print("ERROR: no token found. Set DATAROBOT_API_TOKEN or run `datarobot auth login`.",
              file=sys.stderr)
        sys.exit(1)

    # Normalize: strip /api/v2 suffix so script can always append it consistently
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/api/v2"):
        endpoint = endpoint[: -len("/api/v2")]

    return endpoint, token


def _make_tarball() -> Path:
    # The env installs datarobot-fastrag from PyPI (see install_dependencies.sh),
    # so the Docker context is just the env dir as-is — no wheel to build or inject.
    tmp = Path(tempfile.mkdtemp())
    context = tmp / "context"
    shutil.copytree(ENV_DIR, context)
    tarball = tmp / "context.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(context, arcname=".")
    print(f"  Tarball: {tarball} ({tarball.stat().st_size // 1024} KB)")
    return tarball


def _get_version() -> str:
    import tomllib
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _api(method: str, endpoint: str, token: str, path: str, **kwargs) -> dict:
    url = f"{endpoint.rstrip('/')}{path}"
    resp = requests.request(
        method, url, headers={"Authorization": f"Bearer {token}"}, **kwargs
    )
    if not resp.ok:
        print(f"ERROR: {method} {url} → {resp.status_code}\n{resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp.json()


def _get_or_create_personal_env(endpoint: str, token: str) -> tuple[str, str]:
    """Return (env_id, env_name), creating a private env and caching it if needed."""
    if PERSONAL_ENV_CONFIG.exists():
        cached = json.loads(PERSONAL_ENV_CONFIG.read_text())
        if cached.get("endpoint") == endpoint.rstrip("/"):
            env_id = cached["id"]
            env_name = cached["name"]
            print(f"Personal env: {env_name} ({env_id})")
            return env_id, env_name

    public_info = json.loads(ENV_INFO.read_text())
    name = f"{public_info['name']} [personal]"
    description = (
        f"Personal test copy of '{public_info['name']}' for fastrag development. "
        "Promote to the shared environment via: uv run scripts/upload_dr_env.py --promote"
    )
    print(f"Creating personal environment: {name} ...")
    data = _api(
        "POST", endpoint, token,
        "/api/v2/executionEnvironments/",
        json={
            "name": name,
            "description": description,
            "programmingLanguage": public_info["programmingLanguage"],
        },
        timeout=30,
    )
    env_id = data["id"]
    PERSONAL_ENV_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    PERSONAL_ENV_CONFIG.write_text(json.dumps({
        "id": env_id,
        "name": name,
        "endpoint": endpoint.rstrip("/"),
    }, indent=2))
    print(f"  Created and saved to {PERSONAL_ENV_CONFIG}")
    return env_id, name


def _upload(endpoint: str, token: str, env_id: str, tarball: Path, label: str) -> str:
    print(f"Uploading version {label} ...")
    with open(tarball, "rb") as fh:
        data = _api(
            "POST", endpoint, token,
            f"/api/v2/executionEnvironments/{env_id}/versions/",
            files={"dockerContext": ("context.tar.gz", fh, "application/gzip")},
            data={"label": label, "description": f"fastrag {label}"},
            timeout=120,
        )
    version_id = data["id"]
    print(f"  Version ID: {version_id}")
    return version_id


def _poll(endpoint: str, token: str, env_id: str, version_id: str) -> None:
    print("Waiting for build", end="", flush=True)
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        data = _api("GET", endpoint, token,
                    f"/api/v2/executionEnvironments/{env_id}/versions/{version_id}/",
                    timeout=30)
        status = data.get("buildStatus", "unknown")
        if status == "success":
            print(" done.")
            return
        if status == "failed":
            print(" FAILED.")
            # Try the dedicated build log endpoint first
            try:
                log_data = _api("GET", endpoint, token,
                                f"/api/v2/executionEnvironments/{env_id}/versions/{version_id}/logs/",
                                timeout=30)
                print("Build log:\n", log_data, file=sys.stderr)
            except SystemExit:
                pass
            # Also dump the full version response so we can see all available fields
            import json as _json
            print("Version response fields:", _json.dumps({k: v for k, v in data.items() if k != "buildLog"}, indent=2), file=sys.stderr)
            print("buildLog field:", data.get("buildLog", "(not present)"), file=sys.stderr)
            sys.exit(1)
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)
    print(" timed out.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--promote", action="store_true",
                        help="Upload to the shared public environment (affects all users)")
    args = parser.parse_args()

    endpoint, token = _load_dr_credentials()
    print(f"Endpoint: {endpoint}")

    if args.promote:
        env_info = json.loads(ENV_INFO.read_text())
        env_id = env_info["id"]
        env_name = env_info["name"]
        print(f"PROMOTE mode — target: {env_name} ({env_id})")
        print("This environment is public and visible to ALL users. Ctrl-C to abort.")
        time.sleep(3)
    else:
        env_id, env_name = _get_or_create_personal_env(endpoint, token)

    tarball = _make_tarball()
    version = _get_version()

    version_id = _upload(endpoint, token, env_id, tarball, label=version)
    _poll(endpoint, token, env_id, version_id)

    print(f"\nDone. Version {version} is live on: {env_name}")
    print(f"  {endpoint.rstrip('/')}/account/environments/{env_id}/")


if __name__ == "__main__":
    main()
