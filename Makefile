.PHONY: help install dev test lint format docker-up docker-down migrate generate-data

help:
	@echo "NeuroDispatch - Intelligent Logistics Management System"
	@echo ""
	@echo "Commands:"
	@echo "  install        Install dependencies with Poetry"
	@echo "  dev            Run development server"
	@echo "  test           Run tests"
	@echo "  lint           Run linters"
	@echo "  format         Format code"
	@echo "  docker-up      Start Docker Compose services"
	@echo "  docker-down    Stop Docker Compose services"
	@echo "  migrate        Run database migrations"
	@echo "  generate-data  Generate synthetic data"

install:
	poetry install

dev:
	poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest -v --cov=src --cov-report=html

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

migrate:
	poetry run alembic upgrade head

migrate-create:
	poetry run alembic revision --autogenerate -m "$(MSG)"

generate-data:
	poetry run python -m src.common.data_generator

shell:
	poetry run python -c "from src.common.database import get_session; import asyncio; asyncio.run(get_session())"

