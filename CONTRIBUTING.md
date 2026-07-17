# Contributing

1. Create a virtual environment.
2. Install the repository with `pip install -e ".[dev]"`.
3. Add or update tests for every behavior change.
4. Run `pytest -q` and `ruff check src tests examples` before opening a pull request.

The project intentionally uses small, auditable examples. Changes should preserve numerical
reference checks and transformation-composition tests.
