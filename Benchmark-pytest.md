## Pytest benchmarks (XRADIO)

Opt‑in performance benchmarks for XRADIO using `pytest-benchmark`. Benchmarks are separate from correctness tests and excluded by default.

### What’s included
- Benchmark tests in `tests/unit/measurement_set/test_load_processing_set.py`:
  - Basic load of a processing set
  - Selective load with a time slice
  - Iterator producing the first item
- Marker `benchmark` to opt‑in.
- Default exclusion of benchmarks via `-m "not benchmark"` in `pyproject.toml`.
- Make targets in this directory (`src/xradio/Makefile`) to run and compare benchmarks, and emit histograms.
- Regression policy: fail comparison if mean runtime regresses by 5%.
- Baselines are not committed; store and use JSON files from CI artifacts.

### Manual usage (after building XRADIO)
From `src/xradio/`:

```bash
# Run XRADIO tests (benchmarks are excluded by default)
make test

# Run benchmarks, autosave results (min 3 rounds) and write histogram SVG
make bench
# JSON results are saved under ./.benchmarks/
# Histogram saved to ./benchmarks/xradio-bench.svg

# Compare against a baseline JSON (e.g., downloaded from CI artifacts) and write comparison histogram
make bench-compare BASELINE=.benchmarks/Linux-CPython-3.12-64bit/ci.json
# Histogram saved to ./benchmarks/xradio-compare.svg

# Quick compare against the previous local autosaved run
make bench-compare-last

```

Alternatively, run pytest directly:
```bash
python -m pytest -v tests -m benchmark \
  --benchmark-only \
  --benchmark-min-rounds=3 \
  --benchmark-autosave \
  --benchmark-save=ci \
  --benchmark-histogram=benchmarks/xradio-bench.svg

python -m pytest -v tests -m benchmark \
  --benchmark-only \
  --benchmark-min-rounds=3 \
  --benchmark-compare .benchmarks/<your-baseline>.json \
  --benchmark-compare-fail=mean:5% \
  --benchmark-histogram=benchmarks/xradio-compare.svg

# Compare against the previous autosaved run (shell one-liner)
python -m pytest -v tests -m benchmark \
  --benchmark-only \
  --benchmark-min-rounds=3 \
  --benchmark-compare "$(ls -1t .benchmarks/*/*.json | sed -n '2p')" \
  --benchmark-compare-fail=mean:5% \
  --benchmark-histogram=benchmarks/xradio-compare.svg
```

### Notes
- The `benchmark` marker is defined in `pyproject.toml` and normal runs exclude it.
- Keep input data and environment consistent for stable results.
- CI can upload `.benchmarks/*.json` artifacts and use them for comparisons.
