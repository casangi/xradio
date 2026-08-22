# AGENT.md — XRADIO

Guide for AI coding agents (and humans) working in this repository. Repo root: `/Users/jsteeb/Dropbox/viper_dev/xradio`.

## What is XRADIO

XRADIO (**X**array **Radio** Astronomy **D**ata **IO**) is a pure-Python library that defines and implements xarray-based data schemas for radio-astronomy data. Its production schema is the **Measurement Set v4 (MS v4)**, surfaced on disk as a single `.ps.zarr` **Processing Set** — a 3-level `xarray.DataTree` (Processing Set → MS v4 nodes → metadata sub-datasets) backed by **Zarr v3**. It provides backends to convert CASA Measurement Set v2 (casacore tables) → MS v4, lazily open / eagerly load Processing Sets, read/write **Image** cubes (CASA / FITS / Zarr), and validate data against typed schemas. There is deliberately **no direct MSv2 reader** — you convert once. XRADIO is part of the casangi / VIPER stack (depends on `toolviper` for logging, Dask clients, test-data download, and memory management).

---

## Environment & Setup

> **In THIS workspace** the package is editable-installed into the conda env named **`zinc`** (NOT `xradio`) from `/Users/jsteeb/Dropbox/viper_dev/xradio/src`. Run everything via `conda run -n zinc ...`.

```bash
# Run the full test suite (523 tests collected)
conda run -n zinc python -m pytest

# Unit tests only / stakeholder (integration) tests only
conda run -n zinc python -m pytest tests/unit
conda run -n zinc python -m pytest tests/stakeholder

# Confirm the active on-disk Zarr format (expect 3)
conda run -n zinc python -c "from xradio._utils.zarr.config import ZARR_FORMAT; print(ZARR_FORMAT)"

# Format all code with Black (the only enforced lint)
make python-format     # == black --config pyproject.toml src/ tests/ docs/source/ scripts/
```

### Fresh install (general)

```bash
pip install -e ".[all]"          # editable dev install with every extra
pip install "xradio[zarr]"       # zarr backend only
pip install "xradio[casacore]"   # MSv2->MSv4 conversion + CASA image IO (pulls arcae; Linux+macOS)
```

- **Base `pip install xradio` pulls only `xarray`** → schema-check + JSON export only (no zarr I/O, no conversion).
- Optional extras: `zarr`, `casacore`, `interactive`, `test`, `docs`, `all` (combinable, e.g. `[interactive,casacore,test]`).
- **Primary CASA-table backend: [arcae](https://github.com/ska-sa/arcae)** (Arrow-based casacore table bindings with self-contained wheels for Linux and macOS). Backend selection lives in `src/xradio/_utils/_casacore/backend.py`: arcae first (via the `casacore_from_arcae` / `images_from_arcae` shims that emulate the python-casacore `tables`/`images` API — arcae has no image API, so CASA images are read/written through table access + TaQL), falling back to `python-casacore` if arcae is absent (undocumented backup only — no extra installs it, no workflow uses it). `casatools` is no longer supported.
- `requires-python = ">=3.11, <3.14"` (3.11 / 3.12 / 3.13).

---

## Repository Layout

```
xradio/
├── pyproject.toml            # SOLE packaging/config file. NO [build-system] table (legacy setuptools fallback).
│                             #   version = "v1.2.0"; base dep xarray; extras; [tool.pytest]; [tool.black]
├── Makefile                  # python-format (black), schema-export
├── MANIFEST.in               # sdist include list
├── .readthedocs.yaml         # RTD build (installs docs/sphinx.txt + .[casacore])
├── src/xradio/
│   ├── __init__.py           # ONLY suppresses Zarr v3 warnings (UnstableSpecificationWarning, consolidated-meta). No __version__.
│   ├── measurement_set/      # PRIMARY subsystem: PS / MS v4 IO + MSv2->MSv4 conversion
│   │   ├── __init__.py        #   public API (convert wrapped in try/except for missing casacore)
│   │   ├── convert_msv2_to_processing_set.py
│   │   ├── open_processing_set.py / load_processing_set.py
│   │   ├── processing_set_xdt.py    # .xr_ps accessor (ProcessingSetXdt)
│   │   ├── measurement_set_xdt.py   # .xr_ms accessor (MeasurementSetXdt)
│   │   ├── schema.py                # VisibilityXds, SpectrumXds, sub-dataset & info schemas
│   │   └── _utils/_msv2|_zarr|_utils # internal converters, casacore readers, zarr encoding
│   ├── image/                # Image cube IO (CASA/FITS/Zarr), single Dataset (NOT DataTree)
│   │   ├── __init__.py        #   open_image, load_image, write_image, make_empty_* , ImageXds
│   │   ├── image.py           #   authoritative docstrings
│   │   ├── image_xds.py       #   .xr_img accessor (ImageXds)
│   │   ├── schema.py          #   DataGroupDict only (no full dataclass schema)
│   │   └── _util/{_casacore,_fits,_zarr,...}
│   ├── schema/               # Home-grown xarray-dataclasses reimpl (Data/Coord/Attr + decorators + check.py)
│   ├── _utils/               # zarr config (ZARR_FORMAT), logging, _casacore shim, xarray_helpers, dict_helpers
│   └── testing/              # PUBLIC pytest-free test helpers (assertions, download_*) reused by benchviper
├── tests/
│   ├── unit/                 # mirrors src/ tree; per-area conftest.py
│   └── stakeholder/          # higher-level integration tests
├── docs/source/             # Sphinx docs (RTD); overview.rst, measurement_set/, image_data/, notebooks
└── scripts/export_schema.py # CLI: export VisibilityXds/SpectrumXds to JSON (used by `make schema-export`)
```

---

## Core Data Model

### Processing Set (DataTree)

A **Processing Set** is an `xarray.DataTree` whose **root node** carries `attrs["type"] == "processing_set"`. On disk it is a single `*.ps.zarr` Zarr store with consolidated metadata. Its immediate children are the **MS v4 nodes**.

```
ps_xdt  (root, attrs["type"]=="processing_set")          # accessor: .xr_ps
└── <ms_v4_name>  (MS v4 node)                            # accessor: .xr_ms
    ├── .ds  ==  the MAIN correlated dataset              # VisibilityXds | SpectrumXds
    │            attrs["type"] in {visibility, radiometer, spectrum}
    └── child DataTree nodes (sub-datasets, optional):
        antenna_xds, field_and_source_<group>_xds, pointing_xds,
        weather_xds, system_calibration_xds, gain_curve_xds,
        phase_calibration_xds, phased_array_xds
```

- **MS v4 node = the main dataset itself** (`ms_xdt.ds` is the `VisibilityXds`/`SpectrumXds`). Sub-datasets are **child** nodes (accessed e.g. `ms_xdt.antenna_xds.ds`).
- **`VisibilityXds`** (interferometers): main var `VISIBILITY` (complex64/128), dims `(time, baseline_id, frequency, polarization)`. `type` ∈ `{visibility, radiometer}`.
- **`SpectrumXds`** (single dish): main var `SPECTRUM` (float16/32/64, units Jy), dims `(time, antenna_name, frequency, polarization)`. `type == spectrum`.
- Coords are **eager, lowercase snake_case**; data variables are **lazy, UPPER_SNAKE_CASE**.
- `MSV4_SCHEMA_VERSION = "4.0.0"`. Required attrs on main: `observation_info`, `processor_info`, `data_groups`, `schema_version`, `creator`, `creation_date`.

**Data groups** (`attrs["data_groups"]`): a dict mapping a group name (a `base` group is mandatory) → `{correlated_data, flag, weight, uvw(visibility-only), field_and_source, description, date}`. Lets multiple correlated-data versions (e.g. `VISIBILITY` vs `VISIBILITY_CORRECTED`) coexist, each tied to its own flag/weight/uvw and a specific `field_and_source` child. `get_data_group_name(None)` → `"base"` if present else the first key.

**On-disk node naming (no collisions):**
```
ms_v4_name = pathlib.Path(in_file).name.replace(".ms", "") + "_" + str(ms_v4_id)
```
where `ms_v4_id` is the **zero-padded** partition index (`conversion.py:1392`; `convert_msv2_to_processing_set.py:197`). Embedding the input-MS basename is what lets you append several distinct MSs into one PS without overwriting. ⚠ `.replace(".ms", "")` strips **every** literal `.ms` substring, not just the suffix.

### Image data model

An image is a **single `xr.Dataset`** (no DataTree), `attrs["type"] == "image_dataset"`. Multiple products live as separate UPPERCASE data vars: `SKY`, `MODEL`, `RESIDUAL`, `POINT_SPREAD_FUNCTION`, `PRIMARY_BEAM`, `APERTURE`, `VISIBILITY`, ... Sky cubes use dims `(time, frequency, polarization, l, m)`; aperture/uv cubes `(time, frequency, polarization, u, v)`; `lmuv` images carry both. Beams: `BEAM_FIT_PARAMS_<TYPE>` with dim `beam_params_label = [major, minor, pa]`. Masks: boolean `FLAG_<TYPE>` (note: stored inverted, True = good pixel internally). Organized via `attrs["data_groups"]` (same mechanism as MS, own `DataGroupDict` in `image/schema.py`).

---

## Key Public APIs

### `xradio.measurement_set`

```python
convert_msv2_to_processing_set(
    in_file: str, out_file: str,
    partition_scheme: list = [],                      # extra keys beyond mandatory DDI/pol/obs-mode split:
                                                      #   FIELD_ID, SCAN_NUMBER, STATE_ID, SOURCE_ID,
                                                      #   SUB_SCAN_NUMBER, ANTENNA1. [] = coarsest (use for OTF mosaics)
    partition_filter: Callable[[dict], bool] | None = None,  # predicate keeping matching partitions; RuntimeError if none
    main_chunksize: dict | float | None = None,       # dim->size dict OR target GiB float OR None (one chunk per var)
    with_pointing: bool = True,
    pointing_chunksize: dict | float | None = None,
    pointing_interpolate / ephemeris_interpolate / phase_cal_interpolate / sys_cal_interpolate: bool = False,
    use_table_iter: bool = False,                     # set True for many-row MSv2 with few partitions
    compressor = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="noshuffle"),  # Zarr v3 BytesBytesCodec; blosc-lz4 default (reads ~1.6x faster than zstd)
    add_reshaping_indices: bool = False,
    storage_backend: Literal["zarr","netcdf"] = "zarr",   # "netcdf" is NOT implemented
    parallel_mode: Literal["none","partition","time"] = "none",
    persistence_mode: str = "w-",                     # "w" overwrite | "w-" fail-if-exists (default) | "a" append
) -> None                                             # writes to <out_file>.ps.zarr (.ps.zarr appended if missing)
```
- `parallel_mode`: `none` serial/eager; `partition` wraps each partition in `dask.delayed` then `dask.compute`; `time` requires **exactly one** partition (phased arrays) and chunks along time (degrades to `none` if `main_chunksize` lacks a `time` key). Unrecognized values are coerced to `none` with a warning.

```python
estimate_conversion_memory_and_cores(in_file, partition_scheme=[]) -> (max_GiB, max_cores, suggested_cores)
# max_cores == number of partitions; suggested_cores == ceil(max_cores/4). Under-accounts for sub-xds memory.

open_processing_set(ps_store, scan_intents=None, array_backend="dask") -> xr.DataTree
# LAZY (metadata only; dask-backed). array_backend "dask" (chunks={}) | "xarray" (chunks=None); else ValueError.
# scan_intents filters MS v4s via .xr_ps.query(). Supports s3:// (via s3fs).

load_processing_set(ps_store, sel_parms=None, data_group_name=None,
                    include_variables=None, drop_variables=None, load_sub_datasets=True) -> xr.DataTree
# EAGER (.load() into memory). sel_parms maps ms_v4_name -> isel slice dict.
# load_sub_datasets=False deletes child nodes whose name contains "xds".
```

**`ProcessingSetXdt` — root accessor `.xr_ps`** (guards `attrs["type"]=="processing_set"`, else `InvalidAccessorLocation`):
```python
ps_xdt.xr_ps.summary(data_group_name=None, first_columns=None) -> pandas.DataFrame
#   columns: name, scan_intents, shape, polarization, spw_name, field/source/line names,
#            field_coords, frequencies, *_UID, start/end_frequency  (cached on .meta)
ps_xdt.xr_ps.query(string_exact_match=True, query=None, **col_filters) -> xr.DataTree  # filter MS v4s
ps_xdt.xr_ps.get_ms_xdt()            # returns the single child MS v4 (asserts exactly one)
ps_xdt.xr_ps.get_max_dims() / get_freq_axis() / get_combined_antenna_xds()
ps_xdt.xr_ps.get_combined_field_and_source_xds(data_group_name="base")
ps_xdt.xr_ps.plot_phase_centers() / plot_antenna_positions() / plot_antenna_positions_2d()
```
> ⚠ There is **no** `sel` / `get` / `to_store` method on `.xr_ps`. Filter with `query()`; persist with the standard `DataTree.to_zarr`.

**`MeasurementSetXdt` — per-MS v4 accessor `.xr_ms`** (guards `type` ∈ `{visibility, spectrum, radiometer}`):
```python
ms_xdt.xr_ms.sel(indexers=None, ..., data_group_name=..., **kw) -> xr.DataTree   # xarray label sel + group select
ms_xdt.xr_ms.get_field_and_source_xds(data_group_name=None) -> xr.Dataset
ms_xdt.xr_ms.get_partition_info(data_group_name=None) -> dict
ms_xdt.xr_ms.add_data_group(new_data_group_name, new_data_group={}, data_group_dv_shared_with=None) -> xr.DataTree
ms_xdt.xr_ms.delete_data_variables(variables: list[str]) -> xr.DataTree
```

### `xradio.image`

```python
open_image(store, chunks={}, verbose=False, do_sky_coords=True, selection={}, compute_mask=True) -> xr.Dataset
#   LAZY (dask). Reads CASA / FITS / Zarr (or a dict {image_type: path}). chunks applies to CASA/FITS only.
#   selection: zarr only.  compute_mask: FITS only.  NOTE: named open_image, not read_image.
load_image(store, block_des=None, do_sky_coords=True) -> xr.Dataset    # EAGER; CASA & Zarr only (NOT FITS)
write_image(xds, imagename, out_format="casa", overwrite=False) -> None  # "casa" | "zarr" (FITS write NOT supported)
make_empty_sky_image(phase_center, image_size, cell_size, frequency_coords, pol_coords, time_coords, ...) -> xr.Dataset
make_empty_aperture_image(...) -> xr.Dataset
make_empty_lmuv_image(...) -> xr.Dataset      # carries BOTH lm and uv coords
# accessor: img_xds.xr_img.sel(... data_group_name=...) / add_data_group / delete_data_variables /
#           get_lm_cell_size / add_uv_coordinates / get_uv_in_lambda(freq) / get_reference_pixel_indices
```

### `xradio.schema`
```python
check_dataset(ds, schema, allow_superflous_dims=frozenset()) -> SchemaIssues   # .expect() raises iff issues
check_array(arr, schema) -> SchemaIssues       # dims order-sensitive
check_dict(d, schema) -> SchemaIssues
check_datatree(dt) -> SchemaIssues             # dispatches per-node via attrs["type"]
@schema_checked                                # validate annotated params/return
# Define schemas with @xarray_dataarray_schema / @xarray_dataset_schema / @dict_schema + Data/Coord/Attr annotations.
```

---

## Common Workflows

**Convert one MSv2 → Processing Set, then open & summarize:**
```python
import xarray as xr
import xradio                                   # registers .xr_ps / .xr_ms / .xr_img accessors
from xradio.measurement_set import (
    estimate_conversion_memory_and_cores, convert_msv2_to_processing_set, open_processing_set,
)
import toolviper

msv2_name = "Antennae_North.cal.lsrk.split.ms"
mem, max_cores, suggested_cores = estimate_conversion_memory_and_cores(msv2_name)
viper_client = toolviper.dask.local_client(cores=suggested_cores)   # size the Dask client

convert_msv2_to_processing_set(
    in_file=msv2_name,
    out_file="Antennae_North.cal.lsrk.split.ps.zarr",
    persistence_mode="w",
    parallel_mode="partition",
)

ps_xdt = open_processing_set("Antennae_North.cal.lsrk.split.ps.zarr")
ps_xdt.xr_ps.summary()                          # -> pandas.DataFrame; 'name' column gives MS v4 names
```

**Convert MULTIPLE MSv2 into a single PS (append with `persistence_mode="a"`):**
```python
from xradio.measurement_set import convert_msv2_to_processing_set, open_processing_set

msv2_list = ["small_lofar.ms", "AA2-Mid-sim_00000.ms"]
outfile = "combined_lofar_aa2.ps.zarr"
for msv2 in msv2_list:
    convert_msv2_to_processing_set(
        in_file=msv2, out_file=outfile, parallel_mode="partition", persistence_mode="a",
    )

ps = open_processing_set(outfile)
ps.xr_ps.summary()
```

**Explore / select / compute:**
```python
sub = ps_xdt.xr_ps.query(field_name=[...], scan_name=["17"])     # subset PS
ms_xdt = ps_xdt.xr_ps.query(execution_block_UID="uid://...").xr_ps.get_ms_xdt()
cor_xds = ms_xdt.ds                                              # the VisibilityXds / SpectrumXds
antenna_xds = ms_xdt.antenna_xds.ds                             # a sub-dataset
ms_xdt.isel(frequency=slice(1, 4))                              # standard xarray still works
ms_xdt.ds.VISIBILITY.max().compute()
```

---

## Conversion Internals (brief)

- **Partitioning** (`_utils/_msv2/partition_queries.py::create_partitions`): reads MAIN + FIELD/SOURCE/STATE subtables once into a pandas DataFrame and groupby-aggregates into a list of partition dicts. **Mandatory split keys** are always prepended (`partition_queries.py:46-51`): `[DATA_DESC_ID, OBS_MODE, OBSERVATION_ID, EPHEMERIS_ID]` → data description (= spectral window + polarization setup) and observation mode are always split. User keys appended: `FIELD_ID, SCAN_NUMBER, STATE_ID, SOURCE_ID, SUB_SCAN_NUMBER, ANTENNA1`. Keys absent from the data are dropped before groupby.
- **`parallel_mode` dispatch** (`convert_msv2_to_processing_set.py:198-248`): `partition` wraps each `convert_and_write_partition` in `dask.delayed`, then `dask.compute(delayed_list)`. `time` hard-asserts `len(partitions)==1` (`:164-167`).
- **Per-partition conversion** (`_utils/_msv2/conversion.py::convert_and_write_partition`, `l.1001`): builds a TaQL `WHERE` (`create_taql_query_where`, `l.764`); reshapes flat MSv2 rows into the dense `(time, baseline_id, frequency, polarization)` grid via `tidxs`/`bidxs` (`calc_indx_for_row_split`, `l.390`); assembles `ms_xdt.ds` + sub-xds child nodes (`l.1393-1416`); applies chunking + `add_encoding`; writes via `to_zarr(..., zarr_format=ZARR_FORMAT)` to `out_file/<ms_v4_name>` (`l.1420-1424`).
- **Finalize** (`convert_msv2_to_processing_set.py`): reopens root, sets `attrs["type"]="processing_set"` (`:254`), calls `zarr.consolidate_metadata(...)` (`:255`).
- **casacore read path**: every table-touching module imports `tables` (and `images`/`coordinates` for image IO) from `xradio._utils._casacore.backend`, which selects arcae first and python-casacore as an undocumented fallback. The arcae shim (`_utils/_casacore/casacore_from_arcae.py`) emulates the python-casacore `tables` API: keywords come from `tabledesc()["_keywords_"]` (casacore JsonOut, full precision; NUL padding stripped), and TaQL (`Table.from_taql`, always bound to the open handle via `$1`) covers what arcae lacks — tiled-column reads (`col[blc:trc]` slice expressions, 1-based inclusive), tiled writes (helper-table `UPDATE ... SET col[...] = t2.X`), `CREATE TABLE`, keyword writes (`SET KEYWORD` literals with a JSON `COPY KEYWORD` fallback for `*0`-style field names), table copies and column drops. ⚠ casacore errors inside `from_taql` abort the process (arcae issue), so avoid feeding it invalid TaQL. CASA images (`_utils/_casacore/images_from_arcae.py`) are plain tables: pixels in a single-cell `TiledCellStMan` `map` column, metadata in table keywords; lonpole/latpole are recomputed via astropy/wcslib like casacore's restore does. Tables open read-only (`lockoptions={"option":"usernoread"}`, `ack=False`). `load_generic_table` always injects a TaQL exclusion of `SOURCE_MODEL` (frequently corrupted). The MS test-data generator (`xradio.testing.measurement_set.msv2_io`) builds MSv2 files via arcae's `ms_from_descriptor`/`ms_descriptor` (exposed as `default_ms`/`required_ms_desc`/… on the backend module).

---

## Zarr v3 / Storage

- **`ZARR_FORMAT = 3`** — single source of truth at `src/xradio/_utils/zarr/config.py`. There is **no per-call override and no env var**; it is threaded as `zarr_format=ZARR_FORMAT` into every `to_zarr` / `zarr.open` call. Set it to `2` to revert all writes to Zarr v2 (e.g. for A/B perf comparison).
- **Default compressor / codec:** `zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="noshuffle")` (a v3 `BytesBytesCodec`), set as the converter default at `convert_msv2_to_processing_set.py:69` and `_utils/_msv2/conversion.py:1015`. Chosen over the former `ZstdCodec(level=2)` because blosc-lz4 decompresses ~1.6× faster on high-entropy visibility data (faster loads) for a small ratio cost; `shuffle`/higher levels gave no benefit here (see `dev/LOAD_PERF_FINDINGS.md`). Encoding key is the **plural** `"compressors": (compressor,)` (v3), not v2's singular `"compressor"`.
- **Consolidated metadata:** not part of the v3 spec, but XRADIO still writes (`zarr.consolidate_metadata`) and reads it; `src/xradio/__init__.py` silences the resulting `ZarrUserWarning` and `UnstableSpecificationWarning`.
- `storage_backend="netcdf"` is **not implemented** (only `zarr` works).

### Performance debugging (this repo recently moved to Zarr v3; read/write got slow)

Most likely culprits, in order:
1. **Chunking** — `add_encoding()` (`_utils/_msv2/_zarr/encoding.py`) defaults `chunks` to the **full dim sizes (one chunk per array)** when no chunksize is passed → kills parallel/compressed I/O. Pass `main_chunksize` / `pointing_chunksize`. The image writer (`image/_util/_zarr/xds_to_zarr.py`) additionally enforces a 0.95 GiB max-chunk guard that raises `ValueError`.
2. **Consolidated-metadata asymmetry** in `load_processing_set`: `consolidated=False` on the `sel_parms` path (forces many small per-array metadata reads — slow over S3 / many-file stores) but defaults (attempts consolidated) on the full-store path.
3. **Codec** — flip `ZARR_FORMAT` to `2` to compare against the v2 `numcodecs` path.
4. **File count** — Zarr v3 stores each chunk under a per-array `c/` directory, increasing file count vs v2. (`image/_util/_zarr/xds_from_zarr.py` hard-codes `elif d.name != "c"` to skip these dirs when walking sub-datasets — a v3 assumption.)

Quick survey command:
```bash
grep -rn "zarr_format=ZARR_FORMAT\|consolidate_metadata\|consolidated=" src/xradio --include="*.py" | grep -v __pycache__
```

---

## Testing & Schema

```bash
conda run -n zinc python -m pytest                 # full suite (523 tests; testpaths=[tests], --strict-markers, --import-mode=importlib)
conda run -n zinc python -m pytest tests/unit      # unit only
make schema-export                                 # regenerate schemas/VisibilityXds.json + SpectrumXds.json (PYTHONPATH=src)
```

- **Public testing helpers** live in `xradio.testing` (pytest-free, reused by external projects / benchviper ASV benchmarks):
  - `assert_xarray_datasets_equal(test, true, *, rtol=1e-7, atol=0.0, check_attrs=True, check_encoding=False)`
  - `assert_attrs_dicts_equal(...)`
  - `xradio.testing.measurement_set`: `download_measurement_set(input_ms, directory="/tmp")`, `check_msv4_matches_descr`, `check_processing_set_matches_msv2_descr`
  - `xradio.testing.image`: `download_image`, `download_and_open_image`, `create_empty_test_image`, `assert_image_block_equal`, `remove_path`
- **Test data** downloads via `toolviper.utils.data.download` (default to `/tmp` for MS assets — avoids Dropbox table-locking issues).
- **Schema checking:** validate data with `check_dataset` / `check_array` / `check_dict` / `check_datatree`; call `.expect()` on the returned `SchemaIssues` to raise.
- **CI**: reusable `nrao/gh-actions-templates-public` templates (linux + codecov, macos, integration, basic-schema-install, run-ipynb) plus a Black formatting check (`black.yml`). `cov_project="xradio"`, test path `tests/`.

---

## Writing xarray Accessors — memory-safety rules (MANDATORY)

Background (2026-08 Frontera diagnosis): xarray's *documented* accessor recipe
(`self._obj = xarray_obj` + xarray caching the instance in `obj._cache[name]`)
creates a **reference cycle** `obj → _cache → accessor → obj`. A cycle is
invisible at laptop scale, but it pins the ENTIRE Dataset/DataTree — every
numpy array in it — until a full garbage-collection pass. In the VIPER imaging
pipeline this pinned 1.5–2.5 GB *per mapping task* (superseded image datasets
died mid-task with their accessor cycles attached), ratcheting worker RSS until
nodes ran out of memory. Reference-counted (cycle-free) death is a hard
requirement for XRADIO objects. Reference implementations of the rules below:
`image_xds.py` (`ImageXds`), `measurement_set_xdt.py` (`MeasurementSetXdt`),
`processing_set_xdt.py` (`ProcessingSetXdt`).

1. **Never register an accessor class directly** (no `@xr.register_dataset_accessor("name")`
   on the class). Register a module-level **factory** that constructs the
   instance and immediately weakens its back-reference:
   `xr.register_dataset_accessor("xr_x")(lambda ds: XAccessor(ds)._weaken())`.
2. **Hybrid strong/weak back-reference.** `__init__` stores the object
   strongly (`self._xds_strong = ds; self._xds_ref = None`) so direct
   construction (`XAccessor(ds)` as a standalone wrapper, used in tests and
   templates) keeps wrapper semantics. `_weaken()` — called only by the
   factory — swaps to `weakref.ref`. On the accessor path
   (`ds.xr_x.method()`), `ds` itself keeps the object alive for the duration
   of the call, so the weak reference is always valid when it matters.
3. **Access the object only through the `_xds`/`_xdt` property**, which
   returns the strong reference if set, else dereferences the weakref and
   raises `ReferenceError` with usage guidance if the object is gone. Never
   touch `_xds_strong`/`_xds_ref` from method bodies.
4. **Rebinding must go through a strong local.** Under a weak back-reference,
   `self._xds = self._xds.assign_coords(...)` followed by `return self._xds`
   can lose the new object *between the two statements* (nothing else holds
   it). Always: `xds = self._xds.assign_coords(...); self._xds = xds; return xds`.
   The `_xds` setter preserves the current mode (weak stays weak, strong
   stays strong).
5. **Never store accessor instances** in attributes, containers, or module
   state — use them inline (`ds.xr_x.method()`) and let them die. An accessor
   kept beyond its object's life raises `ReferenceError` by design.
6. **Prove cycle-free death in a unit test** for every new accessor:
   ```python
   ds = make_dataset(); ds.xr_x  # populate the accessor cache
   wr = weakref.ref(ds); del ds
   assert wr() is None            # died by refcount, NO gc.collect() needed
   ```
7. **DataTree caveat (related, not accessor-specific):** `xarray.DataTree`
   parent↔child links are themselves strong reference cycles — any dropped
   tree is cyclic garbage. Consumers that drop task-owned trees must sever
   them (see `astroviper.utils.data_tree.release_data_tree`); XRADIO code
   should avoid keeping dropped subtrees reachable.

---

## Gotchas

- **Accessors require importing their SUBPACKAGE, not just `xradio`.** Bare `import xradio` registers nothing; `import xradio.measurement_set` registers `.xr_ps`/`.xr_ms` and `import xradio.image` registers `.xr_img` (registration is a side effect of the defining module's import). Without it they raise `AttributeError`. Calling a method on the wrong node type raises `InvalidAccessorLocation` (a `ValueError` subclass). `make_empty_*` images set `attrs["type"]="image"` (singular) and the `.xr_img` accessor rejects them until they become an `"image_dataset"`.
- **casacore is optional (but no longer macOS-gated).** `convert_msv2_to_processing_set` / `estimate_conversion_memory_and_cores` are imported inside a `try/except` — if `arcae` is missing (and no python-casacore fallback is present) they emit a `UserWarning` and are simply **absent** from the namespace. `arcae` installs from PyPI on Linux and macOS alike.
- **casacore table locking breaks on Dropbox-backed paths.** Run any casacore/CASA table operations (conversion, CASA-image read/write, related tests) in `$TMPDIR`, not in the Dropbox tree.
- **`storage_backend="netcdf"` is documented but NOT implemented.** Only `zarr` works.
- **No `[build-system]` table** in `pyproject.toml` (legacy setuptools fallback); no `setup.py`/`setup.cfg`. If you add entry points / dynamic version / explicit package discovery, add `[build-system]` first.
- **No `xradio.__version__`.** Version is hardcoded `version = "v1.2.0"` in `pyproject.toml` (note leading `v`; pip normalizes to `1.2.0`).
- **`persistence_mode` defaults to `"w-"`** (fail if `.ps.zarr` exists). Use `"a"` to append, `"w"` to overwrite.
- **Only Black is configured** (line-length 88, target py312) — no ruff/flake8/isort/mypy/pre-commit/tox.
- **`--strict-markers` is on but no custom markers are registered** — `@pytest.mark.<custom>` without registration will error.
- **Two divergent docs dep lists**: the pyproject `docs` extra vs `docs/sphinx.txt` (the latter, pinned `Sphinx==7.2.6` etc., is what RTD actually installs — and RTD installs the `casacore` extra, not `docs`).
- **Notebooks under `docs/source/**/guides` and tutorials are executed by `run-ipynb` CI** — breaking the public API can fail notebook CI even when pytest passes.
- **Image schema is conventional, not formally validated** — `image/schema.py` defines only `DataGroupDict`; image cubes rely on UPPERCASE var names + canonical dims + `attrs["type"]`, with no full dataclass schema.
- **FITS is read-only** (`open_image` reads it; `load_image`/`write_image` do not). Compressed (`CompImageHDU`) or scaled (`BSCALE != 1.0` / `BZERO != 0.0`) FITS are incompatible with memmap and raise `RuntimeError`.
- **`open_processing_set` ≠ `read_image`**; the image reader is `open_image`. No `read_image` exists anywhere.
- **Stray `.DS_Store` files** are committed throughout the tree — ignore them.
- **Two different "schema" things:** the typed dataclass framework in `src/xradio/schema/` vs the unrelated runtime mapper `src/xradio/_utils/schema.py` (`convert_generic_xds_to_xradio_schema`). Don't confuse them.
- **Decorated schema classes are not normal objects** — `@xarray_*_schema` overrides `__new__`, so `MySchema(...)` returns a validated `xr.DataArray`/`Dataset`/`dict`; `isinstance(obj, MySchema)` is `False`. Use `check_dataset` / `check_array`.
- **`xradio_logger()` returns a MODULE** (toolviper's logger module, else stdlib `logging`), not a `Logger` instance — call `.debug(...)` / `.info(...)` directly on it.
