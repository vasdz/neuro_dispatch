.PHONY: help install dev test lint format docker-up docker-down migrate generate-data train metrics

help:
	@echo "NeuroDispatch - Intelligent Logistics Management System"
	@echo ""
	@echo "Development Commands:"
	@echo "  install        Install dependencies with Poetry"
	@echo "  dev            Run development server"
	@echo "  test           Run tests"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  lint           Run linters"
	@echo "  format         Format code"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-up      Start Docker Compose services"
	@echo "  docker-down    Stop Docker Compose services"
	@echo "  docker-logs    View container logs"
	@echo "  docker-build   Build Docker images"
	@echo "  docker-prod    Build production Docker image"
	@echo ""
	@echo "Database Commands:"
	@echo "  migrate        Run database migrations"
	@echo "  generate-data  Generate synthetic data"
	@echo ""
	@echo "MLOps Commands:"
	@echo "  train          Train demand forecasting model"
	@echo "  train-mlflow   Train with MLflow tracking"
	@echo "  tune           Hyperparameter tuning"
	@echo "  metrics        Show current metrics"
	@echo ""
	@echo "Monitoring Commands:"
	@echo "  monitor-up     Start monitoring stack (Prometheus + Grafana)"
	@echo "  monitor-down   Stop monitoring stack"

install:
	poetry install

dev:
	poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest -v

test-cov:
	poetry run pytest -v --cov=src --cov-report=html --cov-report=term-missing

test-mlops:
	poetry run pytest tests/test_mlops.py -v

lint:
	poetry run ruff check src tests
	poetry run mypy src

format:
	poetry run black src tests main.py
	poetry run ruff check --fix src tests

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

docker-prod:
	docker build --target production -t neuro_dispatch:latest -f infra/docker/Dockerfile .

migrate:
	poetry run alembic upgrade head

migrate-create:
	poetry run alembic revision --autogenerate -m "$(MSG)"

generate-data:
	poetry run python -m src.common.data_generator

# MLOps commands
train:
	poetry run python -m src.demand_forecast.training --model lightgbm --days 30

train-mlflow:
	poetry run python -m src.demand_forecast.training --model lightgbm --days 30 --mlflow

tune:
	poetry run python -c "from src.demand_forecast.training import HyperparameterTuner; import asyncio; print('Tuning not implemented in CLI yet')"

metrics:
	@curl -s http://localhost:8000/metrics | head -50

# Monitoring commands
monitor-up:
	docker-compose up -d prometheus grafana

monitor-down:
	docker-compose stop prometheus grafana

health:
	@curl -s http://localhost:8000/health | python -m json.tool

ready:
	@curl -s http://localhost:8000/ready | python -m json.tool

shell:
	poetry run python -c "from src.common.database import get_session; import asyncio; asyncio.run(get_session())"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

