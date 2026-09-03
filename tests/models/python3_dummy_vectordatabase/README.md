## Python Dummy Vector Database Inference Model Template

A minimal vector database model. Expects a `promptText` column in the input dataset and
returns, per row, a list of retrieved documents in the `relevant` target column plus
citation metadata columns that DataRobot surfaces as extra model output.

## Instructions
Create a new custom model with this `custom.py` and use any Python Drop-In Environment with it.
`TARGET_NAME` must be set to `relevant`, matching the target column returned by `score`.
