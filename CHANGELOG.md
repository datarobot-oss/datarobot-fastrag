# Changelog

All notable changes to `datarobot-fastrag` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- python-semantic-release inserts generated sections directly below the flag on the
     next line (`changelog.insertion_flag`), so keep it above the newest entry. -->

<!-- version list -->

## v0.2.14 (2026-11-12)

- `GET /models` now returns an OpenAI-shaped list, so `openai` `models.list()` works, and falls back to the `LLM_ID` runtime parameter when the `get_supported_llm_models` hook is missing.
- `POST /predictions` for text generation models now returns the `target_name` column as the prediction when the hook adds extra columns (for example moderation guards), instead of the first column.

## v0.2.13 (2026-11-10)

- Raise the declared requirement `aiohttp>=3.13.3` to `aiohttp>=3.14.4` to match a CVE fix this repo
  already enforces.
- Raise the declared requirement `openai>=2.20.0` to `openai>=2.30.0` to match a CVE fix this repo
  already enforces.
- Raise the declared requirement `opentelemetry-api>=1.23.0` to `opentelemetry-api>=1.45.1` to match
  a CVE fix this repo already enforces.
- Raise the declared requirement `pydantic-settings>=2.7.1` to `pydantic-settings>=2.14.2` to match
  a CVE fix this repo already enforces.
- Raise the declared requirement `pytest>=9.0.2` to `pytest>=9.0.3` to match a CVE fix this repo
  already enforces.
- Raise the declared requirement `python-multipart>=0.0.9` to `python-multipart>=0.0.31` to match a
  CVE fix this repo already enforces.

## v0.2.12 (2026-11-06)

- Minimum Python version required bumped to 3.13, minimum uv version to 0.13.4.
- Added a set of dependency constraints to help fight CVEs.

## v0.2.11 (2026-10-20)

- Fixed scoring with an async `score` hook and a moderation pipeline failing with `'coroutine' object has no attribute 'columns'` by awaiting score hook on the server event loop

## v0.2.10 (2026-10-16)

- Fixed issue where non-streaming chat with a moderation pipeline failed with "Cannot run the event loop while another loop is running" by moving the moderation library's custom metric uploads outside event loop.
- moderation pipeline now receives datarobot_association_id (from the request or generated if not provided)
- Reported each chat's prompt/response row with its association ID to DataRobot (`predictionInputs/fromJSON`), this fixes Data exploration's Data quality and Data export features

## v0.2.9 (2026-10-09)

- Fixed chat streams that fail mid-response by ending them with a generic `server_error` SSE event and `data: [DONE]`, and reported them to MLOps monitoring stats as system errors instead of successes

## v0.2.8 (2026-10-02)

- Passed a chat hook just the kwargs it declares when a moderation pipeline is loaded, so moderated chat no longer raises `unexpected keyword argument` for hooks that omit `target_type` or `headers`
- Bumped `datarobot-moderations` to 11.3.7

## v0.2.7 (2026-09-23)

- Forwarded request headers to `chat` and `score` as a case-insensitive `headers` kwarg
- Passed a chat hook just the kwargs it declares when no moderation pipeline is loaded, fixes issue `unexpected keyword argument 'target_type'` for LLM blueprint plus vector database RAG custom models

## v0.2.6 (2026-09-09)

- Allowed `GET /models` (`get_supported_llm_models`) for the `agenticworkflow` target type, matching the chat route

## v0.2.5 (2026-09-08)

- Added local container memory profiler (`make mem-profile`)
- Added version bump command (`make bump`)

## v0.2.4 (2026-09-03)

- Fixed `vectordatabase` target type predictions being returned as objects instead of lists
- Stripped double quotes from `TARGET_NAME` environment variable if present

## v0.2.3 (2026-09-02)

- Renamed leftover references of project's former name (FastDRUM) to FastRAG
- Added `changelog-check` CI

## v0.2.2 (2026-08-19)

- Chat-completion prediction stats reporting for deployments
- Dropped `datarobot-mlops` from example and test execution environments

## v0.2.1 (2026-08-11)

- First public release
