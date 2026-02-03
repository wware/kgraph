#!/bin/bash -e

fixes_needed() {
    echo "Something needs fixing, trying to fix it"
    set -x
    uv run black kgraph kgbundle kgserver
    uv run ruff check --fix kgraph kgbundle kgserver
    exit 1
}

mypy_fix_needed() {
    echo "Something mypy-ish needs fixing, You need to do that"
    exit 1
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

PYTHONFILES=$(git ls-files -- kgraph kgbundle kgschema kgserver examples | grep -E '\.py$')

echo ""
echo "=========================================="
echo "Running ruff check..."
echo "=========================================="
uv run ruff check ${PYTHONFILES} || fixes_needed

echo ""
echo "=========================================="
echo "Running mypy..."
echo "=========================================="
uv run mypy ${PYTHONFILES} || mypy_fix_needed

echo ""
echo "=========================================="
echo "Running black check..."
echo "=========================================="
uv run black --check ${PYTHONFILES} || fixes_needed

echo ""
echo "=========================================="
echo "Running flake8..."
echo "=========================================="
uv run flake8 ${PYTHONFILES} --count --show-source --statistics || fixes_needed

echo ""
echo "=========================================="
echo "Running pylint..."
echo "=========================================="
uv run pylint ${PYTHONFILES}

echo ""
echo "=========================================="
echo "Running tests..."
echo "=========================================="
# Run tests from root for kgraph and examples
uv run pytest tests/ -q

# Run kgbundle tests (if they exist)
if [ -d "kgbundle/tests" ]; then
    echo ""
    echo "Running kgbundle tests..."
    (cd kgbundle && uv run pytest tests/ -q)
fi

# Run kgserver tests with kgbundle in PYTHONPATH
if [ -d "kgserver/tests" ]; then
    echo ""
    echo "Running kgserver tests..."
    ROOT_DIR="$(pwd)"
    (cd kgserver && PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}" uv run pytest tests/ -q)
fi
