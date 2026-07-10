.PHONY: install test bench lint format docs clean doctor

# ── Development Setup ────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# ── Testing ──────────────────────────────────────────────────────────
test:
	pytest tests/ -q

test-verbose:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=cognicore --cov-report=term-missing

# ── Benchmarks ───────────────────────────────────────────────────────
bench:
	python -m cognicore.benchmarks.suite --quick

bench-full:
	python -m cognicore.benchmarks.suite --output benchmark_output/latest.json --csv benchmark_output/latest.csv

bench-regression:
	python -m cognicore.benchmarks.regression --baseline benchmark_output/baseline.json

# ── Code Quality ─────────────────────────────────────────────────────
lint:
	ruff check cognicore/ tests/

format:
	ruff format cognicore/ tests/

# ── Diagnostics ──────────────────────────────────────────────────────
doctor:
	python -m cognicore.cli doctor

# ── Cleanup ──────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
