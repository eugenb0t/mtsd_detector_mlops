.PHONY: sync lint format test cov up-db up-mlflow up-prod down pre-commit smoke env mlflow-pipeline

sync:
	uv sync

env:
	@test -f .env || cp .env.example .env
	@echo ".env ready"

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest -q

cov:
	uv run pytest --cov=mtsd_detector --cov-report=term-missing --cov-fail-under=80

up-db: env
	docker compose up -d postgres

up-mlflow: env
	docker compose up -d mlflow

mlflow-pipeline:
	uv run mlflow-pipeline --tracking-uri $${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}

up-prod: env
	docker compose --profile prod up --build -d

down:
	docker compose --profile prod down

pre-commit:
	uv run pre-commit run --all-files

smoke:
	@curl -sf http://127.0.0.1:8000/healthz | grep -q '"status":"ok"'
	@curl -sf http://127.0.0.1:8000/api/v1/version | grep -q '"version"'
	@curl -sf http://127.0.0.1:8000/api/v1/health | grep -q '"status"'
	@echo "smoke ok"
