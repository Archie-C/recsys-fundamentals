PYTHON := $(if $(wildcard venv/bin/python),venv/bin/python,python3)

.PHONY: help data test lint docker

help:
	@echo "Available targets:"
	@echo "  make data   - download & verify MovieLens-100K"
	@echo "  make test   - run pytest suite"
	@echo "  make lint   - run ruff/black"
	@echo "  make docker - build Docker image"

data:
	${PYTHON} get_data.py

test:
	${PYTHON} -m pytest -q

lint:
	ruff check src tests

docker:
	docker build -t recsys-sprint:latest .
