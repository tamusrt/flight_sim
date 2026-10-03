---
name: run-flight-sim-checks
description: "Run the project's formatting, linting, type-checking, and test checks. Use when validating changes or preparing a pull request."
---

# Run Flight Sim Checks

Run these commands from the repository root:

1. `uv run --frozen ruff check .`
2. `uv run --frozen ruff format --check .`
3. `uv run --frozen pylint $(git ls-files '*.py')`
4. `uv run --frozen mypy .`
5. `uv run --frozen pytest`

Report each command's result and include test coverage when available.