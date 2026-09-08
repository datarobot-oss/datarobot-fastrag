## Python Dummy Text Generation Chat Model Template

This is a simple text generation model that supports OpenAI API chat() and models().

## Instructions
Create a new custom model with this `custom.py` and use any GenAI Python Drop-In Environment with it.

### To run locally with `fastrag`
Paths are relative to the repository root. `fastrag server` has no target-type flag, so it
comes from an environment variable (or a `model-metadata.yaml` in the code dir):

```bash
TARGET_TYPE=textgeneration fastrag server \
  --code-dir tests/models/python3_dummy_chat \
  --address localhost:6789
```

### Using `curl`:

#### List models:
```bash
curl localhost:6789/models
```

#### Chat:
```bash
curl -X POST http://localhost:6789/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Greetings"}]}'
```

### Using OpenAI Python client:

Note: `client.models.list()` does not work against `fastrag`. The `/models` route returns the
bare list from the `get_supported_llm_models` hook, not the `{"object": "list", "data": [...]}`
envelope the OpenAI client expects, so use the `curl` above to list models.

#### Simple chat:

```python
from openai import OpenAI

url = "http://localhost:6789"
api_token = "not-needed"
client = OpenAI(base_url=url, api_key=api_token, _strict_response_validation=False)

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[ {'role': 'user', 'content': 'Greetings'} ],
    temperature=0,
)
print(response)
```
