#!/bin/bash
set -e

# Minimal execution environment for datarobot-fastrag.
#
# datarobot-fastrag is installed from the local wheel below and pulls its full
# required dependency closure automatically (datarobot-moderations core, fastapi,
# uvicorn, pydantic, openai, opentelemetry, numpy, pandas, tiktoken, rouge-score,
# nltk, ...). The heavy "batteries" (torch, transformers, faiss, onnx, langchain,
# llama-index, nemoguardrails, vector-DB clients, ...) and the DataRobot MLOps
# Java monitoring agent are intentionally NOT installed. Add per-model deps to
# requirements.txt.

# No system build tools: the entire required closure installs from prebuilt
# manylinux wheels (verified on ubi9/python-312-minimal), so gcc/g++ are not
# needed. No java-openjdk / nginx either -- this env runs `fastrag server`
# (uvicorn) directly, with no MLOps Java agent and no nginx proxy.
microdnf update -y
microdnf clean all

pip3 install -U pip --no-cache-dir
pip3 install --no-cache-dir wheel setuptools

# Extra per-model deps (if any) declared in requirements.txt.
pip3 install -r requirements.txt --no-cache-dir --upgrade-strategy eager

# datarobot-fastrag + its full required closure (incl. datarobot-moderations core).
pip3 install --no-cache-dir datarobot_fastrag-0.2.1-py3-none-any.whl

rm -f requirements.txt
rm -f datarobot_fastrag-0.2.1-py3-none-any.whl
