#!/usr/bin/env bash
# Builds a local test Docker image from the current source, starts it with a
# minimal test model, and verifies the server responds correctly under concurrent
# load. Specifically exercises the asyncio event loop fix in SyncModelAdapter.
#
# Usage: ./scripts/verify_docker.sh [--keep]
#   --keep  Leave the container and image running after the test (for manual inspection)
#
# Prerequisites: Docker running, base image pulled:
#   docker pull datarobotdev/buzok-genai-custom-model-local-dropin-env:latest

set -euo pipefail

BASE_IMAGE="datarobotdev/buzok-genai-custom-model-local-dropin-env:latest"
IMAGE_NAME="fastrag-local-test"
CONTAINER_NAME="fastrag-verify"
PORT=8085
MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/tests/local_test_model"
KEEP=false

for arg in "$@"; do
  [[ "$arg" == "--keep" ]] && KEEP=true
done

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
info() { echo -e "  ${YELLOW}→${NC} $1"; }

FAILED=0

cleanup() {
  if [[ "$KEEP" == false ]]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
  else
    echo ""
    info "Container left running (--keep): docker logs $CONTAINER_NAME"
  fi
}
trap cleanup EXIT

echo ""
echo "fastrag Docker verification"
echo "──────────────────────────────────────"

# ── 1. Build wheel ────────────────────────────────────────────────────────────
info "Building wheel..."
uv build --wheel -q
WHEEL=$(ls dist/datarobot_fastrag-*.whl | sort -V | tail -1)
pass "Wheel built: $(basename "$WHEEL")"

# ── 2. Build image ────────────────────────────────────────────────────────────
info "Building Docker image (from $BASE_IMAGE)..."
docker build \
  -f Dockerfile.local-test \
  -t "$IMAGE_NAME" \
  --quiet \
  . >/dev/null
pass "Image built: $IMAGE_NAME"

# ── 3. Start container ────────────────────────────────────────────────────────
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
info "Starting container on port $PORT..."
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "${PORT}:8080" \
  -v "${MODEL_DIR}:/opt/model" \
  -e TARGET_TYPE=regression \
  --entrypoint fastrag \
  "$IMAGE_NAME" \
  server --code-dir /opt/model --address 0.0.0.0:8080 \
  >/dev/null

# ── 4. Wait for readiness ─────────────────────────────────────────────────────
info "Waiting for server to be ready..."
BASE_URL="http://localhost:${PORT}"
for i in $(seq 1 30); do
  if curl -sf "${BASE_URL}/ping/" >/dev/null 2>&1; then
    pass "Server ready (${i}s)"
    break
  fi
  if [[ $i -eq 30 ]]; then
    fail "Server did not become ready after 30s"
    echo ""
    echo "Container logs:"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -20
    exit 1
  fi
  sleep 1
done

echo ""
echo "Endpoint checks"
echo "──────────────────────────────────────"

# ── 5. /ping/ ─────────────────────────────────────────────────────────────────
PING=$(curl -sf "${BASE_URL}/ping/")
if echo "$PING" | grep -q '"ok"'; then
  pass "GET /ping/  → $PING"
else
  fail "GET /ping/  → unexpected: $PING"
  FAILED=1
fi

# ── 6. /health/ ───────────────────────────────────────────────────────────────
HEALTH=$(curl -sf "${BASE_URL}/health/")
if echo "$HEALTH" | grep -q '"ok"'; then
  pass "GET /health/ → $HEALTH"
else
  fail "GET /health/ → unexpected: $HEALTH"
  FAILED=1
fi

# ── 7. /info/ ─────────────────────────────────────────────────────────────────
INFO=$(curl -sf "${BASE_URL}/info/")
if echo "$INFO" | grep -q '"server":"fastrag"'; then
  pass "GET /info/  → server=fastrag, target_type=regression"
else
  fail "GET /info/  → unexpected: $INFO"
  FAILED=1
fi

# ── 8. Single POST /predict/ ─────────────────────────────────────────────────
PRED=$(curl -sf -X POST "${BASE_URL}/predict/" \
  -H "Content-Type: text/csv" \
  --data-binary $'feature1,feature2\n1.0,2.0\n3.0,4.0\n5.0,6.0')
COUNT=$(echo "$PRED" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['predictions']))" 2>/dev/null || echo 0)
if [[ "$COUNT" == "3" ]]; then
  pass "POST /predict/ (3 rows) → $COUNT predictions"
else
  fail "POST /predict/ (3 rows) → got count=$COUNT, body=$PRED"
  FAILED=1
fi

echo ""
echo "Concurrency check (asyncio event loop fix)"
echo "──────────────────────────────────────"

# ── 9. 30 concurrent requests ────────────────────────────────────────────────
info "Sending 30 concurrent requests..."
CONC_RESULT=$(python3 - <<'EOF'
import urllib.request, json, threading

PORT = 8085
URL  = f"http://localhost:{PORT}/predict/"
DATA = b"feature1,feature2\n" + b"\n".join(f"{i},{i+1}".encode() for i in range(10))

results, errors, lock = [], [], threading.Lock()

def run():
    try:
        req = urllib.request.Request(URL, data=DATA, headers={"Content-Type": "text/csv"})
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read())
            with lock:
                results.append(len(body["predictions"]))
    except Exception as e:
        with lock:
            errors.append(str(e))

threads = [threading.Thread(target=run) for _ in range(30)]
for t in threads: t.start()
for t in threads: t.join()

print(f"ok={len(results)} errors={len(errors)}")
if errors:
    for e in errors[:3]:
        print(f"  error: {e}")
EOF
)

OK_COUNT=$(echo "$CONC_RESULT" | grep -o 'ok=[0-9]*' | cut -d= -f2)
ERR_COUNT=$(echo "$CONC_RESULT" | grep -o 'errors=[0-9]*' | cut -d= -f2)

if [[ "$OK_COUNT" == "30" && "$ERR_COUNT" == "0" ]]; then
  pass "30/30 concurrent requests succeeded, 0 errors"
else
  fail "Concurrent requests: $CONC_RESULT"
  FAILED=1
fi

# ── 10. Check logs for RuntimeError ──────────────────────────────────────────
RUNTIME_ERRORS=$(docker logs "$CONTAINER_NAME" 2>&1 | grep -c "RuntimeError" || true)
if [[ "$RUNTIME_ERRORS" == "0" ]]; then
  pass "No RuntimeError in container logs"
else
  fail "Found $RUNTIME_ERRORS RuntimeError(s) in logs"
  docker logs "$CONTAINER_NAME" 2>&1 | grep "RuntimeError"
  FAILED=1
fi

echo ""
echo "──────────────────────────────────────"
if [[ "$FAILED" == "0" ]]; then
  echo -e "${GREEN}All checks passed.${NC}"
else
  echo -e "${RED}Some checks failed.${NC}"
  exit 1
fi
