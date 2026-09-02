.PHONY: install-deps
install-deps: ## Install core dependencies via uv
	pip install uv
	uv sync

.PHONY: install-deps-integration
install-deps-integration: ## Install core + integration dependencies via uv
	pip install uv
	uv sync --group integration

.PHONY: build
build: ## Build the wheel
	uv build

.PHONY: publish
publish: publish-pypi-prod ## Alias for publish-pypi-prod

.PHONY: publish-pypi-test
publish-pypi-test: build ## Build wheel and publish to Test PyPI
	@echo "Publishing datarobot-fastrag to Test PyPI"
	UV_PUBLISH_TOKEN=$(PYPI_TOKEN) uv publish --publish-url https://test.pypi.org/legacy/

.PHONY: publish-pypi-prod
publish-pypi-prod: build ## Build wheel and publish to PyPI
	@echo "Publishing datarobot-fastrag to PyPI"
	UV_PUBLISH_TOKEN=$(PYPI_TOKEN) uv publish
	./scripts/tag_and_push_release.sh

.PHONY: clean
clean: ## Removes build/test artifacts and caches
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf `find . -name __pycache__`
	rm -rf `find . -type f -name '*.py[co]' `
	rm -rf `find . -type f -name '*~' `
	rm -rf `find . -type f -name '.*~' `
	rm -rf `find . -type f -name '@*' `
	rm -rf `find . -type f -name '#*#' `
	rm -rf `find . -type f -name '*.orig' `
	rm -rf `find . -type f -name '*.rej' `
	rm -rf .coverage
	rm -rf coverage
	rm -rf build
	rm -rf htmlcov
	rm -rf dist

.PHONY: fmt
fmt:
	@echo "Running code formatters"
	@uv run ruff format .
	@uv run ruff format tests

fix:
	@uv run ruff check --select I --fix tests .
	@uv run ruff check . --fix

.PHONY: cov
cov:
	@uv run pytest -s -v  --cov-report term --cov-report html --cov fastrag ./tests
	@echo "open file://`pwd`/htmlcov/index.html"


.PHONY: test
test:
	@uv run pytest -svvv tests -m "not integration"

.PHONY: integration-test
integration-test: ## Run integration tests (requires datarobot-moderations installed)
	@uv run pytest -svvv tests -m "integration"


.PHONY: mypy
mypy:
	@uv run mypy --pretty --strict fastrag


.PHONY: lint
lint:
	@uv run ruff check .

.PHONY: changelog-check
changelog-check: ## Verify CHANGELOG.md documents the version in pyproject.toml
	@python3 scripts/check_changelog.py

ci: lint mypy fmt test

.PHONY: verify
verify: ## Build a local Docker image and run endpoint + concurrency checks
	@bash scripts/verify_docker.sh

.PHONY: upload
upload: ## Build wheel and upload new execution environment version to DataRobot SaaS
	@uv run scripts/upload_dr_env.py

.PHONY: test-models
test-models: ## Correctness test: baseline on current env then fastrag env (pass --model-id, --dataset-id, etc.)
	@uv run scripts/test_models.py correctness $(ARGS)
