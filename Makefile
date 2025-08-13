 .PHONY: python-format test bench bench-compare bench-compare-last

LATEST_BENCH := $(shell ls -1t .benchmarks/*/*.json 2>/dev/null | head -n1)
PREV_BENCH   := $(shell ls -1t .benchmarks/*/*.json 2>/dev/null | sed -n '2p')

# Format Python code using black
python-format:
	black --config pyproject.toml src/ tests/ docs/source/

# Run xradio tests (excluding benchmarks by default via pyproject)
test:
	python -m pytest -v tests

# Run benchmarks only, autosave results, and use 3 min rounds
bench:
	mkdir -p benchmarks
	python -m pytest -v tests -m benchmark --benchmark-only \
	  --benchmark-min-rounds=3 --benchmark-autosave --benchmark-save=ci \
	  --benchmark-histogram=benchmarks/xradio-bench.svg

# Compare current benchmarks against a baseline JSON, failing on 5% regression
# Usage: make bench-compare BASELINE=.benchmarks/Linux-CPython-3.12-64bit/ci.json
bench-compare:
	mkdir -p benchmarks
	python -m pytest -v tests -m benchmark --benchmark-only \
	  --benchmark-min-rounds=3 --benchmark-compare $(BASELINE) \
	  --benchmark-compare-fail=mean:5% \
	  --benchmark-histogram=benchmarks/xradio-compare.svg

# Compare against the previous autosaved run in .benchmarks/
bench-compare-last:
	@if [ -z "$(PREV_BENCH)" ]; then \
	  echo "Not enough benchmark runs found under .benchmarks to compare."; \
	  echo "Run 'make bench' at least twice, or use 'make bench-compare BASELINE=...'"; \
	  exit 1; \
	fi
	mkdir -p benchmarks
	python -m pytest -v tests -m benchmark --benchmark-only \
	  --benchmark-min-rounds=3 --benchmark-compare $(PREV_BENCH) \
	  --benchmark-compare-fail=mean:5% \
	  --benchmark-histogram=benchmarks/xradio-compare.svg
