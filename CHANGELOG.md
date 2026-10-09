# Changelog

All notable changes to `datarobot-fastrag` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- python-semantic-release inserts generated sections directly below the flag on the
     next line (`changelog.insertion_flag`), so keep it above the newest entry. -->

<!-- version list -->

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
