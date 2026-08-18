# datarobot-fastrag

FastRAG is an async-native rewrite of [DRUM](https://github.com/datarobot/datarobot-user-models) built on FastAPI. It serves DataRobot custom models using the same `custom.py` hook interface as DRUM, but handles each request on an async event loop instead of blocking a thread. For I/O-bound LLM workloads (vector DB + LLM calls), this typically gives 3–5× higher throughput at the same concurrency level.

Existing DRUM `custom.py` files work without modification — sync hooks are run in a thread pool automatically.

## Features

- `async def chat()` / `async def score()` run natively on the event loop
- Sync hooks offloaded to a `ThreadPoolExecutor` (drop-in compatible with existing models)
- OpenAI-compatible chat completions API (`/v1/chat/completions`, streaming included)
- Predict/score API (`/predict/`)
- OpenTelemetry instrumentation built in
- Deployment prediction stats (Total Predictions and errors) reported to DataRobot
- LLM safety guardrails via [datarobot-moderations](https://pypi.org/project/datarobot-moderations/)

### Not implemented
- Artifact guessing
- Transform and custom tasks API
- `directAccess` routes

## Installation

FastRAG requires Python 3.12 or later.

```bash
pip install datarobot-fastrag
```

Or using uv:

```bash
uv add datarobot-fastrag
```

For local development from source:

```bash
git clone https://github.com/datarobot-oss/datarobot-fastrag
cd datarobot-fastrag
uv pip install -e .
```

## Writing a custom LLM model

FastRAG uses the same `custom.py` hook convention as DRUM. For LLM models, you implement `chat()`. Making it `async` is what unlocks the throughput benefit.

### Step 1 — create your model directory

```
my_model/
├── custom.py
└── model-metadata.yaml
```

`model-metadata.yaml`:

```yaml
name: My LLM model
type: inference
targetType: textgeneration
```

### Step 2 — implement `custom.py`

The minimal interface for a chat model:

```python
# custom.py
import httpx

async def load_model(code_dir: str):
    # Return anything — it is passed as `model` to every hook.
    # Initialise your clients here (LLM, vector DB, etc).
    return httpx.AsyncClient()

async def chat(completion_create_params: dict, model, **kwargs):
    messages = completion_create_params["messages"]
    user_prompt = next(m["content"] for m in reversed(messages) if m["role"] == "user")

    # Both awaits release the event loop, so other requests run concurrently.
    # db_results = await model.post("http://vector-db/search", json={"q": user_prompt})
    # answer = await model.post("http://llm/generate", json={"prompt": ...})

    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": completion_create_params["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": f"Echo: {user_prompt}"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
```

### Step 3 — run locally

```bash
fastrag server --code-dir ./my_model
```

Test it:

```bash
curl -X POST http://localhost:8080/v1/chat/completions/ \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "messages": [{"role": "user", "content": "hello"}]}'
```

Or with the OpenAI Python client:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = await client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "hello"}],
)
print(response.choices[0].message.content)
```

### Step 4 — deploy on DataRobot

Upload `custom.py` and `model-metadata.yaml` as a custom model in the DataRobot UI, and select the **[GenAI] Python 3.12 with Moderations** execution environment. When the `GENAI_RAG_FRAG_RUNNER` platform flag is enabled for your org, the environment starts FastRAG automatically; otherwise it falls back to DRUM.

### All supported hooks

| Hook | Signature | Notes |
|---|---|---|
| `load_model` | `(code_dir: str) -> Any` | Return value becomes `model` in all other hooks. Runs once at startup. |
| `init` | `(code_dir: str) -> None` | Side-effecting setup (logging, connections). Runs before `load_model`. |
| `chat` | `(completion_create_params: dict, model: Any, **kwargs) -> dict \| Iterator` | OpenAI chat completions. Return a dict or a (sync/async) generator for streaming. |
| `score` | `(data: pd.DataFrame, model: Any, **kwargs) -> pd.DataFrame` | Tabular predictions. |
| `score_unstructured` | `(data: Any, model: Any, **kwargs) -> Any` | Raw bytes in/out. |
| `get_supported_llm_models` | `(model: Any) -> list[Model]` | Populates `/v1/models`. |

All hooks can be `async def` or plain `def`. Sync hooks run in a thread pool.

### Streaming

Return a generator from `chat()` that yields `ChatCompletionChunk` objects (or plain dicts). Both sync generators and `async` generators work:

```python
async def chat(completion_create_params, model, **kwargs):
    if completion_create_params.get("stream"):
        async def gen():
            for token in ["Hello", " world"]:
                yield {"choices": [{"delta": {"content": token}, "finish_reason": None, "index": 0}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}
        return gen()
    # non-streaming fallback ...
```

A complete working example (including streaming) is at [`tests/models/python3_dummy_chat/custom.py`](tests/models/python3_dummy_chat/custom.py).

## Configuration

Configuration can be provided via CLI arguments or environment variables:

| CLI Argument | Environment Variable | Description |
|--------------|----------------------|-------------|
| `--code-dir` | `CODE_DIR` | Directory containing `custom.py` and model files. |
| `--address` | `ADDRESS` | Host and port to bind to (e.g., `0.0.0.0:8080`). |
| `--max-workers` | `MAX_WORKERS` | Number of worker processes. |
| `--verbose` | `VERBOSE` | Enable verbose logging. |

### Prediction stats reporting

FastRAG reports prediction counts to DataRobot using the same env vars as DRUM
(`EXTERNAL_WEB_SERVER_URL` / `API_TOKEN` / `DEPLOYMENT_ID` / `MODEL_ID`; `MLOPS_*`
aliases are accepted). Records are queued and POSTed in batches; reporting never
blocks a request. Chat counts as 1, `/predict/` counts one per row, and 4xx/5xx
are reported as `userError` / `systemError` with 0 predictions. Reporting is off
only when those credentials are missing (typical for local runs).

## Development

The project uses `uv` for dependency management and a `Makefile` for common tasks.

To run the tests:

```bash
make test
```

To run with coverage:

```bash
make cov
```

To format the code:

```bash
make fmt
```

To clean up build and test artifacts:

```bash
make clean
```
