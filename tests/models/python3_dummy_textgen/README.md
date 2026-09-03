## Python Dummy Text Generation Inference Model Template

This text generation model is a very simple model that generates text output based on input.
It works with any Python environment that has `pandas`.
Expects `input` column name in the input dataset to have text. Output results are reversed text inputs.

## Instructions
Create a new custom model with this `custom.py` and use any Python Drop-In Environment with it.

### To run locally with `fastrag`
Paths are relative to the repository root. `fastrag server` has no target-type or class-label
flags, so those come from environment variables (or a `model-metadata.yaml` in the code dir):

```bash
TARGET_TYPE=textgeneration fastrag server \
  --code-dir tests/models/python3_dummy_textgen \
  --address localhost:6789
```

To submit a request using `curl`:

```bash
curl -X POST http://localhost:6789/predictions/ \
  -H "Content-Type: text/csv" \
  --data-binary $'input\nhello world'
```

This model reads an `input` column, which `tests/data.csv` does not have, so the
request above sends the CSV inline.
