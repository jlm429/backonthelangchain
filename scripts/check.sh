#!/usr/bin/env bash
set -euo pipefail

echo "Running Ruff lint..."
poetry run ruff check .

echo "Running tests..."
poetry run pytest

echo "All checks passed."