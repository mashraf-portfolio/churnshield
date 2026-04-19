.PHONY: install lint format test pre-commit docker-build docker-up docker-down clean

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest --cov=src --cov-report=term-missing tests/

pre-commit:
	pre-commit run --all-files

docker-build:
	docker build --target api -t churnshield-api:local .
	docker build --target dashboard -t churnshield-dashboard:local .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
