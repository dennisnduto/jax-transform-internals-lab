.PHONY: install test lint demo

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests examples

demo:
	python -m jax_transform_internals.demo
