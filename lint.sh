#!/bin/bash -e

fixes_needed() {
    echo "Something needs fixing, trying to fix it"
    set -x
    sed -i "s/ \+$//" $(git ls-files | grep -E "\.py$")
    uv run ruff check --fix . && uv run black .
}

echo "=========================================="
echo "Running Linters and Tests"
echo "=========================================="

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Please install uv first."
    echo "See: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo ""
echo "UV Version:"
uv --version

echo ""
echo "=========================================="
echo "Running ruff check..."
echo "=========================================="
uv run ruff check . || fixes_needed

echo ""
echo "=========================================="
echo "Running mypy..."
echo "=========================================="
uv run mypy . || fixes_needed

echo ""
echo "=========================================="
echo "Running black check..."
echo "=========================================="
uv run black --check . || fixes_needed

echo ""
echo "=========================================="
echo "Running flake8..."
echo "=========================================="
uv run flake8 . --count --show-source --statistics --exclude=.venv || fixes_needed

echo ""
echo "=========================================="
echo "Running pylint..."
echo "=========================================="
uv run pylint $(find . -name "*.py" | grep -v venv)

echo ""
echo "=========================================="
echo "Running tests..."
echo "=========================================="
uv run pytest tests/ -v
