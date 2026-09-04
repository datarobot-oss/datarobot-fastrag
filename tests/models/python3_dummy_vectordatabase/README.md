## Python Dummy Vector Database Inference Model Template

A minimal vector database model. Expects a `promptText` column in the input dataset and
returns, per row, a list of retrieved documents in the `relevant` target column plus
citation metadata columns that DataRobot surfaces as extra model output.

## Instructions
Create a new custom model with this `custom.py` and use any Python Drop-In Environment with it.

### To run locally with `fastrag`
Paths are relative to the repository root. `fastrag server` has no target-type or class-label
flags, so those come from environment variables (or a `model-metadata.yaml` in the code dir):

```bash
TARGET_TYPE=vectordatabase \
  TARGET_NAME=relevant \
  fastrag server \
  --code-dir tests/models/python3_dummy_vectordatabase \
  --address localhost:6789
```

To submit a request using `curl`:

```bash
curl -X POST http://localhost:6789/predictions/ \
  -H "Content-Type: text/csv" \
  --data-binary @tests/data.csv
```

`TARGET_NAME` names the column holding the retrieved documents; every other column
is returned as `extraModelOutput`.
