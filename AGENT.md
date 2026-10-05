# AGENT.md — XRADIO

Guide for AI coding agents (and humans) working in this repository. Repo root: `/Users/jsteeb/Dropbox/viper_dev/xradio`.

## What is XRADIO

XRADIO (**X**array **Radio** Astronomy **D**ata **IO**) is a pure-Python library that defines and implements xarray-based data schemas for radio-astronomy data. Its production schema is the **Measurement Set v4 (MS v4)**, surfaced on disk as a single `.ps.zarr` **Processing Set** — a 3-level `xarray.DataTree` (Processing Set → MS v4 nodes → metadata sub-datasets) backed by **Zarr v3**. It provides backends to convert CASA Measurement Set v2 (casacore tables) → MS v4, lazily open / eagerly load Processing Sets, read/write **Image** cubes (CASA / FITS / Zarr), and validate data against typed schemas. An MSv2 is either converted once (`convert_msv2_to_processing_set`) or opened lazily, without converting it, as the same Processing Set with the **`xradio_msv2` xarray engine** (`open_msv2`, see "MSv2 xarray engine" below), which stores its partitions inside the MS. XRADIO is part of the casangi / VIPER stack (depends on `toolviper` for logging, Dask clients, test-data download, and memory management).

---

## Environment & Setup

> **In THIS workspace** the package is editable-installed into the conda env named **`zinc`** (NOT `xradio`) from `/Users/jsteeb/Dropbox/viper_dev/xradio/src`. Run everything via `conda run -n zinc ...`.

```bash
# Run the full test suite
conda run -n zinc python -m pytest

# Unit tests only / stakeholder (integration) tests only
conda run -n zinc python -m pytest tests/unit
conda run -n zinc python -m pytest tests/stakeholder

# Confirm the active on-disk Zarr format (expect 3)
conda run -n zinc python -c "from xradio._utils.zarr.config import ZARR_FORMAT; print(ZARR_FORMAT)"

# Lint and format with ruff (the version pinned in .pre-commit-config.yaml)
uvx ruff@0.12.5 check --fix <files> && uvx ruff@0.12.5 format <files>
pre-commit run --all-files   # every hook CI runs (ruff, pyupgrade, absolufy-imports, hygiene)
```

Run tests and scripts that write CASA tables from a directory outside Dropbox
(for example `$TMPDIR`), see Gotchas.

### Fresh install (general)

```bash
pip install -e ".[all]"          # editable dev install with every extra
pip install "xradio[zarr]"       # zarr backend only
pip install "xradio[casacore]"   # MSv2->MSv4 conversion + CASA image IO (python-casacore, not on macOS)
```

- **Base `pip install xradio` pulls only `xarray`** → schema-check + JSON export only (no zarr I/O, no conversion).
- Optional extras: `zarr`, `casacore`, `interactive`, `test`, `docs`, `all` (combinable, e.g. `[interactive,casacore,test]`).
- **CASA-table backends:** MSv2→MSv4 conversion and CASA image IO use **python-casacore** when it is importable, else **casatools** through the shim `xradio._utils._casacore.casacore_from_casatools` (macOS, where the `casacore` extra skips python-casacore). FITS and zarr images need neither (FITS I/O uses astropy only).
- `requires-python = ">=3.11, <3.14"` (3.11 / 3.12 / 3.13).
- **Editable installs must be reinstalled** (`pip install -e . --no-deps`) after a change to `pyproject.toml` entry points (for example the xarray backends), because the installed `entry_points.txt` is only written at install time.

---

## Repository Layout

```
xradio/
├── pyproject.toml            # SOLE packaging/config file. NO [build-system] table (legacy setuptools fallback).
│                             #   version = "v1.2.4"; base dep xarray; extras; xarray backend entry points;
│                             #   [tool.pytest]; [tool.ruff]
├── .pre-commit-config.yaml   # hooks CI runs (pre-commit.yml): ruff 0.12.5 check + format, pyupgrade, absolufy-imports, hygiene
├── Makefile                  # python-format (legacy Black target), schema-export
├── MANIFEST.in               # sdist include list
├── .readthedocs.yaml         # RTD build (installs docs/sphinx.txt + .[casacore])
├── schemas/                  # JSON exports of VisibilityXds, SpectrumXds, ImageXds (`make schema-export`)
├── src/xradio/
│   ├── __init__.py           # ONLY suppresses Zarr v3 warnings (UnstableSpecificationWarning, consolidated-meta). No __version__.
│   (src/_xradio_xarray_backends.py, a top level module next to xradio/: xarray backend
│    entry points (CASA/FITS images, MSv2), imports only os + xarray.backends, so loading
│    them never imports xradio)
│   ├── measurement_set/      # PRIMARY subsystem: PS / MS v4 IO + MSv2->MSv4 conversion
│   │   ├── __init__.py        #   public API (convert wrapped in try/except for missing casacore)
│   │   ├── convert_msv2_to_processing_set.py
│   │   ├── open_processing_set.py / load_processing_set.py
│   │   ├── processing_set_xdt.py    # .xr_ps accessor (ProcessingSetXdt)
│   │   ├── measurement_set_xdt.py   # .xr_ms accessor (MeasurementSetXdt)
│   │   ├── schema.py                # VisibilityXds, SpectrumXds, sub-dataset & info schemas
│   │   └── _utils/_msv2|_zarr|_utils # internal converters, casacore readers, zarr encoding
│   │                                  #   (_msv2/open_msv2.py: open_msv2, remove_msv2_partition_cache, the xradio_msv2 engine)
│   ├── image/                # Image cube IO (CASA/FITS/Zarr), single Dataset (NOT DataTree)
│   │   ├── __init__.py        #   open_image, load_image, write_image, make_empty_*, check_image, ImageXds (accessor)
│   │   ├── image.py           #   authoritative docstrings
│   │   ├── image_xds.py       #   .xr_img accessor (ImageXds)
│   │   ├── schema.py          #   ImageXds dataset schema, array/dict schemas, DataGroupDict, check_image
│   │   ├── backends.py        #   xarray backends implementation (open_image_dataset, lazy BackendArray)
│   │   └── _util/
│   │       ├── conventions.py #     shared name tables: spectral frames, FITS Stokes codes, image types, time
│   │       ├── _write_plan.py #     write_image output planning, naming and staged (atomic) writes
│   │       ├── legacy.py      #     upgrade of image zarr stores written by xradio <= 1.2.3
│   │       ├── _blocks.py     #     RegionReader: batched dask computation of the writers' pixel regions
│   │       └── _casacore/, _fits/, _zarr/, casacore.py, fits.py, zarr.py, image_factory.py, common.py
│   ├── schema/               # Home-grown xarray-dataclasses reimpl (Data/Coord/Attr + decorators + check.py)
│   │   └── measures.py        #   shared measure schemas (time, spectral, sky, location, ...), also
│   │                          #   re-exported by xradio.measurement_set.schema for backward compatibility
│   ├── _utils/               # zarr config (ZARR_FORMAT), logging, _casacore shim, xarray_helpers, dict_helpers
│   └── testing/              # PUBLIC pytest-free test helpers (assertions, download_*) reused by benchviper
├── tests/
│   ├── unit/                 # mirrors src/ tree; per-area conftest.py
│   ├── stakeholder/          # higher-level integration tests
│   └── casatools/            # MSv2 reads/conversions/engine with real casatools (skipped with python-casacore)
├── docs/source/             # Sphinx docs (RTD); overview.rst, measurement_set/, image_data/, notebooks
└── scripts/export_schema.py # CLI: export a dataset schema to JSON (used by `make schema-export`)
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
where `ms_v4_id` is the **zero-padded** partition index (padded in `convert_msv2_to_processing_set`, joined in `conversion.py::_convert_and_write_partition`). Embedding the input-MS basename is what lets you append several distinct MSs into one PS without overwriting. ⚠ `.replace(".ms", "")` strips **every** literal `.ms` substring, not just the suffix.

### Image data model

An image is a **single `xr.Dataset`** (no DataTree), `attrs["type"] == "image_dataset"`. Multiple products live as separate UPPERCASE data vars: `SKY`, `POINT_SPREAD_FUNCTION`, `PRIMARY_BEAM`, `MASK`, `APERTURE`, `VISIBILITY`, ..., with versions named by suffix (`SKY_MODEL`, `SKY_RESIDUAL`, `FLAG_SKY_RESIDUAL`). Sky cubes use dims `(time, frequency, polarization, l, m)`; aperture/uv cubes `(time, frequency, polarization, u, v)`; `lmuv` images carry both. Beams: `BEAM_FIT_PARAMS_<TYPE>` with dim `beam_params_label = [major, minor, pa]`. Flags: boolean `FLAG_<TYPE>`, `True` = invalid pixel (CASA masks, where `True` = good, are inverted on read and write). Organized via `attrs["data_groups"]` (same mechanism as MS): each group maps roles (`sky`, `flag`, `point_spread_function`, `primary_beam`, `mask`, ...) to variable names.

- **Formal schema:** `xradio.image.schema.ImageXds` (dataset schema; not the `xradio.image.ImageXds` accessor class) with array and dict schemas, exported to `schemas/ImageXds.json`. `check_image(xds)` = `check_dataset` + data group roles + the `flag`/`beam_fit_params` references of images + beam labels; it does not check that a writer can store the dataset (non-uniform axes, icrs/hcrs/lsr spectral observers, polarization sets FITS cannot hold).
- **Conventions** (one translation table each in `image/_util/conventions.py`, used by both readers, both writers and the factories): the frequency coordinate's `frame` holds the casacore name (`LSRK`, `BARY`, `TOPO`, `GEO`, `REST`, `LSRD`, `GALACTO`, `LGROUP`, `CMB`), `reference_frequency.attrs.observer` the schema vocabulary (`lsrk`, `BARY`, `gcrs`, ...), FITS `SPECSYS` its FITS name; polarization labels are kept in **canonical Jones order** (`I, Q, U, V`; `RR, RL, LR, LL`; `XX, XY, YX, YY`) in memory, so correlations map onto 2x2 Jones matrices: the FITS writer reorders the planes into a linear FITS `STOKES` axis, every reader (FITS, CASA, zarr) returns the canonical order (CASA images converted from FITS can store another one), the `make_empty_*` factories raise for another order and `check_image` reports it; `sub_type` uses casacore image types without spaces (`SpectralIndex`), written back in casacore's spelling; times are read through their `units`, `format` and `scale`.
- `l`/`m` are projection plane coordinates (pixel offset x cdelt; direction cosines for SIN), `u`/`v` are in wavelengths (`units: "lambda"`). Frequencies are in Hz; `frequency.attrs.channel_width` (as in MSv4) gives the width of a single-channel axis.

---

## Key Public APIs

### `xradio.measurement_set`

```python
convert_msv2_to_processing_set(
    in_file: str, out_file: str,
    partition_scheme: list | None = None,             # None = []; extra keys beyond mandatory DDI/pol/obs-mode split:
                                                      #   FIELD_ID, SCAN_NUMBER, STATE_ID, SOURCE_ID,
                                                      #   SUB_SCAN_NUMBER, ANTENNA1. [] = coarsest (use for OTF mosaics)
    partition_filter: Callable[[dict], bool] | None = None,  # predicate keeping matching partitions; RuntimeError if none
    main_chunksize: dict | float | None = None,       # dim->size dict OR target GiB float OR None: time-only chunks of
                                                      #   ~128 MiB (largest data var, >= 1 time step); {} = one chunk per var
    with_pointing: bool = True,
    pointing_chunksize: dict | float | None = None,
    pointing_interpolate / ephemeris_interpolate / phase_cal_interpolate / sys_cal_interpolate: bool = False,
    use_table_iter: bool = False,                     # DEPRECATED no-op (True emits a DeprecationWarning)
    compressor = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="noshuffle"),  # Zarr v3 BytesBytesCodec; blosc-lz4 default (reads ~1.6x faster than zstd)
    add_reshaping_indices: bool = False,
    storage_backend: Literal["zarr","netcdf"] = "zarr",   # "netcdf" is NOT implemented
    parallel_mode: Literal["none","partition","time"] = "none",
    persistence_mode: str = "w-",                     # "w" overwrite | "w-" fail-if-exists (default) | "a" append
) -> None                                             # writes to <out_file>.ps.zarr (.ps.zarr appended if missing)
```
- `parallel_mode`: `none` serial; `partition` wraps each partition in `dask.delayed` then `dask.compute`; `time` requires **exactly one** partition (phased arrays) and, only with a `time` key in `main_chunksize`, reads the large MAIN columns lazily (one dask block per time chunk; any row order, missing/duplicated `(time, baseline)` rows give the `none` output). Without a `time` key (also the default `None`) `time` converts as `none` (streamed write) and logs a warning saying to pass `main_chunksize={"time": n}`. `time` still fails at write time on a column with undefined or variable-shape cells, which `none`/`partition` skip. Unrecognized values are coerced to `none` with a warning.
- **`main_chunksize=None`** (`conversion.py::default_main_chunksize`): chunks along time only, ~128 MiB (`DEFAULT_MAIN_CHUNK_BYTES`) of the largest data variable, at least one time step. Only a time step over the Blosc limit (2 GiB) also splits frequency, then baseline, with a WARNING (the streamed write still holds one whole time step per batch). `main_chunksize={}` gives the former default (one chunk per variable, fails over 2 GiB with Blosc).
- **Streamed write** (`_utils/_msv2/stream_write.py`; `none`, `partition`, and `time` without a `time` key): the MAIN data variables are lazy placeholders (`deferred_placeholder`, raise if computed); `DataTree.to_zarr(compute=False, consolidated=False)` writes all metadata and the other variables, then `write_deferred_variables` fills one variable at a time in batches of whole time chunks (`STREAM_BATCH_BYTES` = 128 MiB, at least one time chunk, so one time step when a step is larger; up to 8x for fragmenting row orders), byte-identical to one `to_zarr`. The MSv4's consolidated metadata is written last (`consolidate_msv4`): an interrupted write leaves an MSv4 that opens with a warning (fails with `consolidated=True`) and that a new PS root (mode `w`/`w-`) does not list; in mode `a`, re-converting an existing MSv4 name, the PS root still lists it (as on main). Partitions whose data variables are <= 1/4 batch are read whole and written by one `to_zarr`. A column read failing mid-write (`DeferredReadError`) or zarr metadata the batch writer cannot apply (`DeferredEncodingError`) removes the MSv4 (`discard_msv4`, also URLs via fsspec) and `convert_and_write_partition` converts the partition again (without that column / without streaming; `w-` becomes `w` only when the failed attempt created the MSv4, so a pre-existing MSv4 is never overwritten).

```python
estimate_conversion_memory_and_cores(in_file, partition_scheme=[]) -> (max_GiB, max_cores, suggested_cores)
# max_cores == number of partitions; suggested_cores == ceil(max_cores/4). Under-accounts for sub-xds memory.

open_processing_set(ps_store, scan_intents=None, array_backend="dask") -> xr.DataTree
# LAZY (metadata only; dask-backed). array_backend "dask" (chunks={}) | "xarray" (chunks=None); else ValueError.
# scan_intents filters MS v4s via .xr_ps.query(). Supports s3:// (via s3fs).

open_msv2(ms_path, scan_intents=None, array_backend="dask", **engine_kwargs) -> xr.DataTree
# = xr.open_datatree(ms_path, engine=MSv2BackendEntrypoint, chunks={} | None, **engine_kwargs): the PS that
# convert_msv2_to_processing_set would write, opened as open_processing_set opens it, WITHOUT converting.
# engine_kwargs: partition_scheme, partition_filter, main_chunksize, with_pointing, pointing_chunksize,
# *_interpolate (as the converter), drop_variables, skip_columns (MAIN columns treated as unreadable, as the
# converter treats a failed read), partition_cache ("auto"|"read"|"rebuild"|"off", default
# $XRADIO_MSV2_PARTITION_CACHE or "auto"), on_partition_error ("skip" default | "raise").
# Errors exported by xradio.measurement_set: MSv2ChangedError, MSv2ReadError, PartitionCacheWarning.
remove_msv2_partition_cache(ms_path) -> bool   # removes the MAIN keyword, then the XRADIO_PARTITIONS sub-table (xradio's only)

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
open_image(store, chunks=None, verbose=False, do_sky_coords=True, selection=None, compute_mask=True) -> xr.Dataset
#   LAZY (dask). Reads CASA / FITS / Zarr; store is a path, a dict {image_type: path} or a list of paths
#   (image types detected from the names: the last dot token when it names a role, else the role
#   substrings xradio 1.2.4 used, e.g. target_psf.im; outputs of a multi data group write all end in
#   .sky, so reopen them with a dict). chunks: CASA/FITS only. selection: zarr only (an int keeps
#   the dim). compute_mask: FITS only. FITS and zarr need no casacore. Zarr stores written by
#   xradio <= 1.2.3 are upgraded on read (image/_util/legacy.py). NOTE: open_image, not read_image.
load_image(store, block_des=None, do_sky_coords=True) -> xr.Dataset    # EAGER; CASA & Zarr only (NOT FITS)
write_image(xds, imagename, out_format="casa", overwrite=False) -> None  # "casa" | "fits" | "zarr"
make_empty_sky_image(phase_center, image_size, cell_size, frequency_coords, pol_coords, time_coords, ...) -> xr.Dataset
make_empty_aperture_image(...) -> xr.Dataset
make_empty_lmuv_image(...) -> xr.Dataset      # carries BOTH lm and uv coords
#   make_empty_* datasets have type "image_dataset" and an empty "base" data group.
check_image(xds) -> SchemaIssues               # image schema check (also xradio.image.schema.check_image)
# accessor: img_xds.xr_img.sel(... data_group_name=...) / add_data_group / delete_data_variables /
#           get_lm_cell_size / add_uv_coordinates / get_uv_in_lambda(freq) / get_reference_pixel_indices
xr.open_dataset(path, engine="xradio_casa_image" | "xradio_fits_image", chunks=..., image_type=...,
                image_chunks=..., drop_variables=...)   # engines also inferred from the path
```

- **`write_image` outputs:** zarr writes the whole dataset to one store. CASA and FITS images hold one image each: every data variable that is an image of some data group is written once (flags and beam fit parameters travel with their image). One image is written to `imagename`; several to `<imagename>.<g1>...<gn>.<role>`, where `g1..gn` are, in data group order, all data groups whose `<role>` refers to the variable (`out.base.sky`, `out.dirty.residual.primary_beam`); for FITS a `.fits` extension stays last (`img.fits` gives `img.base.sky.fits`). FITS holds only sky plane images; CASA skips the normalization images (no l/m or u/v) with a warning. The CASA and FITS writers raise for non-uniform frequency, l/m or u/v axes. Both compute the pixels region by region (whole dask chunks) through `image/_util/_blocks.py::RegionReader`, which materializes the task graph once and runs batches of regions (up to `BATCH_BYTES`) on the active dask scheduler from culled subgraphs: never call `dask.compute` per region (that culls the whole graph each time, quadratic in the number of chunks). Errors name the dataset variable, not the internal SKY/APERTURE (`_write_plan.naming_the_variable`).
- **Staged writes** (`image/_util/_write_plan.py`): every output path is planned first; with `overwrite=False` any existing output raises `FileExistsError` before anything is written. Outputs are written into a hidden sibling `.<name>.<random>.writing` directory and moved into place only after all succeeded (existing outputs replaced only with `overwrite=True`; other files, such as outputs of earlier writes with other names, are left alone). On failure the staging directory is removed and nothing at the target changes, so a lazily opened image can be written back onto its own path. `~` is expanded and missing parent directories are created.
- **xarray backends:** the entry points live in the light top level module `_xradio_xarray_backends` (`src/_xradio_xarray_backends.py`, outside the `xradio` package: it imports only `os` and `xarray.backends`, so engine-less `xr.open_dataset` calls import neither xradio, whose `__init__` imports zarr and sets zarr warning filters, nor xradio.image); the FITS engine claims files whose primary HDU holds an image; the implementation is `xradio.image.backends.open_image_dataset`: lazily indexed `BackendArray`s that read only the needed planes (one-plane read blocks, `encoding["preferred_chunks"]`), plus `image_type`, `image_chunks` and `drop_variables` parameters. Each engine opens only its own format.

### `xradio.schema`
```python
check_dataset(ds, schema, allow_superflous_dims=frozenset()) -> SchemaIssues   # .expect() raises iff issues
check_array(arr, schema) -> SchemaIssues       # dims order-sensitive
check_dict(d, schema) -> SchemaIssues
check_datatree(dt) -> SchemaIssues             # dispatches per-node via attrs["type"] (incl. "image_dataset")
@schema_checked                                # validate annotated params/return
# Define schemas with @xarray_dataarray_schema / @xarray_dataset_schema / @dict_schema + Data/Coord/Attr annotations.
# Shared measure schemas (TimeArray, SpectralCoordArray, SkyCoordArray, ...) live in xradio.schema.measures.
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

- **Partitioning** (`_utils/_msv2/partition_queries.py::create_partitions_with_main_rows`): reads the MAIN key columns + FIELD/SOURCE/STATE subtables once into a pandas DataFrame and groupby-aggregates into a list of partition dicts; it also returns every partition's MAIN rows as row runs (`MainRowRuns`, one vectorized group-by: the rows `create_taql_query_where` would select), passed to each `convert_and_write_partition`. **Mandatory split keys** are always prepended (`_create_partitions`): `[DATA_DESC_ID, OBS_MODE, OBSERVATION_ID, EPHEMERIS_ID]` → data description (= spectral window + polarization setup) and observation mode are always split. User keys appended: `FIELD_ID, SCAN_NUMBER, STATE_ID, SOURCE_ID, SUB_SCAN_NUMBER, ANTENNA1`. Keys absent from the data are dropped before groupby.
- **`parallel_mode` dispatch** (`convert_msv2_to_processing_set.py`): `partition` wraps each `convert_and_write_partition` in `dask.delayed`, then `dask.compute(delayed_list)`. `time` hard-asserts `len(partitions)==1`.
- **Per-partition conversion** (`_utils/_msv2/conversion.py::convert_and_write_partition`, a retrying wrapper with the signature of `_convert_and_write_partition`): **no TaQL per partition on MAIN** (`create_taql_query_where` only builds the `taql_where` string kept in `partition_info`): opens the base MAIN table with the partition's row runs (`open_partition_main_table` → `_tables/read_rows.py::MainTableRows`); reshapes the rows into the dense `(time, baseline_id, frequency, polarization)` grid via `tidxs`/`bidxs` (`calc_indx_for_row_split`); reads every column with `read_rows_to_grid` (ascending rows in one pass, bounded `getcolnp`/`getcolslicenp` calls of <= 2**26 elements and >= a few MiB each, scattered rows through a <= 16 MiB temporary, never the whole column; the casatools shim reads with bounded `getcol` calls); assembles `ms_xdt.ds` + sub-xds child nodes; applies `default_main_chunksize` (or `main_chunksize`) + `add_encoding`; writes `out_file/<ms_v4_name>` (streamed write, or one `to_zarr`, see above).
- **Sub-table cache** (`_tables/subtable_cache.py`, python-casacore only): one `SubtableCache` per conversion, shared by its partitions: POINTING read once and selected per partition in numpy (`_tables/read_pointing.py`), sorted time columns for the time-range projections, memoized `load_generic_table` results of sub-tables loaded with identical arguments (bounded LRU, deep copies). Pickled to dask workers as a token: the copies in one process share one state, kept 30 s after the last task (at most 2). Output identical to the uncached reads; the MS must not change during a conversion.
- **Finalize** (`convert_msv2_to_processing_set.py`): reopens root, sets `attrs["type"]="processing_set"`, calls `zarr.consolidate_metadata(...)`, after the last partition: the PS root of an interrupted conversion lists none of its new MSv4s.
- **casacore read path**: table-touching modules do `from casacore import tables` and fall back to `xradio._utils._casacore.casacore_from_casatools` (a casatools shim with python-casacore's API) when python-casacore is missing (casatools tables must not be used from several threads at once: every lazy read of a casatools table, by the image and MSv2 readers and the `xradio_msv2` engine, holds the one process-wide `xradio._utils._casacore.tables.CASATOOLS_LOCK` through `casatools_serialized()`, a null context with python-casacore; `_tables/read.py::_CASATOOLS_READ_LOCK` is an alias). CASA images are read through `casacore.images` (or the shim); FITS images only through astropy. Tables open read-only (`lockoptions={"option":"usernoread"}`, `ack=False`). `SOURCE_MODEL` (frequently corrupted) is never read. The MS test-data generator (`xradio.testing.measurement_set.msv2_io`) builds MSv2 files with the same table backends.

---

## MSv2 xarray engine (`xradio_msv2`)

`xr.open_datatree(ms, engine="xradio_msv2", chunks={})` (or `open_msv2`) gives the processing set that `convert_msv2_to_processing_set(ms, ...)` would write, opened with `open_processing_set`, without converting: same node names, variables, values, attributes (build dates aside), and under `chunks={}` the converter's dask chunks of the main data variables. Pass the engine **class** (`_xradio_xarray_backends.MSv2BackendEntrypoint`) where entry points are not installed (editable installs before a reinstall, the zinc env).

- **Files** (`_utils/_msv2/` unless noted):
  - `src/_xradio_xarray_backends.py::MSv2BackendEntrypoint`: the light entry class (stdlib + `xarray.backends`); `_is_measurement_set` guess (`table.info` type, or an empty type with the ANTENNA/DATA_DESCRIPTION/POLARIZATION/SPECTRAL_WINDOW sub-tables); imports the reader only on open.
  - `backend_open.py::open_msv2_tree`: the driver (scheme validation, partitions from `partition_cache`, `partition_filter` with the converter's numbering, one node per partition, `on_partition_error`, the >1,000-partition warning, one retry with recomputed partitions on `StalePartitionsError`).
  - `backend_partition.py::open_partition`: `conversion.build_partition(..., defer_main_columns=True)`, then the placeholders swapped for lazy arrays, the reversed-frequency SPWs reversed lazily, `preferred_chunks` from the converter's encoding, `drop_variables`.
  - `backend_arrays.py`: `MSv2MainColumnArray` (OUTER indexing: only the rows of the selected times and baselines; reads whole cells with the converter's `read_grid` in time sub-blocks of <= 128 MiB; opens and closes MAIN per read; checks MAIN nrows; when the data managers of `ROW_KEY_COLUMNS` were written since the open (`keys_token`) or MAIN is open for writing in the process, checks the rows it reads: still in their partition (`rows_left`: the grouping keys of `partition_grouping`, else `MSv2ChangedError`) and with the TIME/ANTENNA1/ANTENNA2 of the index (else the index is rebuilt)), `OnesArray` (WEIGHT=1), `PartitionIndex` (row runs + a blake2b token of the time/baseline grid + the grouping keys; per-row index in `INDEX_MEMO`, seeded at open, rebuilt with `calc_indx_for_row_split` elsewhere, 64 striped rebuild locks). Its base class is adapted from the ASDM backend's (BASIC there; TODO: unify once both are merged).
  - `backend_pointing.py`: the lazy `pointing_xds` (`read_pointing_index`: TIME/ANTENNA_ID of any number of rows, the cell shapes of every array column whose description does not fix them (`getcolshapestring` in chunks, no values; the common shape, and the `odd_rows` of other shapes or without a value) and one cell per data column read at open; values read per selection with the converter's selection, first-row dedup and `pivot_time_antenna`, rows checked against the index; partitions whose rows include `odd_rows`, and tables the index cannot describe: `PointingBuildArray`, built by the converter's code at open and again on read, one build per partition shared by its variables in `POINTING_BUILD_MEMO`; eager only with `pointing_interpolate=True`).
  - `partition_cache.py`: the `XRADIO_PARTITIONS` sub-table (read and write protocols, staleness rules, `PARTITIONS_MEMO`, the modes); `_tables/table_lock_file.py`: parses casacore's `table.lock` (row counts, data-manager counters) without opening tables, for the fingerprint.
  - `backend_errors.py` (`MSv2ChangedError`, `MSv2ReadError`, `PartitionCacheWarning`, internal `StalePartitionsError` and `MainRowsChangedError`); public API in the private module `_utils/_msv2/open_msv2.py`, exported by `xradio.measurement_set` (as ASDM's `open_asdm`: `xradio.measurement_set.open_msv2` is the function, there is no such submodule); equality helper `xradio/testing/measurement_set/equivalence.py`.
- **Shared build:** the converter's `_convert_and_write_partition` is the `build_partition` context manager (yields a `BuiltPartition`) plus the writer. Any change to the build changes both the converter and the engine: converter output must stay byte-identical (A/B against the previous converter), and the engine equality tests must pass.
- **The converter never reads nor writes the partition cache.**
- **Cache invariants:**
  - One row per (`SCHEME_KEY`, `ALGORITHM_VERSION`), at most 8; no Int64 anywhere (casatools 6.7 cannot read it: counts and run arrays are Double, read back with an integrality check); a CHECKSUM over normalised values.
  - Stale unless (L1) the fingerprint (MAIN data managers, key-column data-manager counters and files, FIELD/STATE/SOURCE) is unchanged and (L2) HISTORY has no rows newer than xradio's own (by row number). L1 and L2 are checked before the row is parsed. Partitions served from the stored row or the memo are verified against their MAIN rows (`verify_partition_rows`); a mismatch drops them and opens once more with recomputed partitions: logged at INFO if the MS changed since they were validated (`changed_since`: fingerprint or HISTORY count), else a "please report" `PartitionCacheWarning`.
  - Write order: sub-table at a temporary name, renamed, then the MAIN keyword (never a keyword without its directory), then the HISTORY row, then the row (CHECKSUM last). Identical recomputed content refreshes the anchor only, without a HISTORY row.
  - Bump `partition_queries.PARTITION_ALGORITHM_VERSION` whenever the output of `create_partitions*` changes (the tripwire test in `test_partition_queries.py` fails otherwise), and `partition_cache.FORMAT_VERSION` for a layout change.
  - Never stored with casatools (the shim cannot create tables), nor in read-only MSs, MSs locked by another process, reference/concatenated MAINs (checked before the lock file, which they lack), MAINs write-locked by this process or MAINs with other data managers: computed in memory, with a `PartitionCacheWarning` or an INFO line once per MS and reason.
  - Never `resync()` a table this process holds the write lock of (`table_lock_file.resync_unless_write_locked`): it drops the process's changes not flushed (e.g. a user's added rows). A MAIN whose rows in the process differ from the lock file while write-locked here gives partitions computed from the process's rows (`compute_in_memory`), without the cache.
  - `remove_msv2_partition_cache` renames the sub-table to a temporary name under the MAIN write lock and deletes it afterwards, so that a concurrent first store cannot link the sub-table being deleted.
- **Lock rules:**
  - casatools: hold `CASATOOLS_LOCK` (`casatools_serialized()`) over open, read, close and release of every table the engine reads lazily, and over each partition build and partition computation.
  - Cache writes: inside the per-MS in-process `write_mutex` (an RLock that also covers the cache's own reads: python-casacore's `close()` unlocks the table object a process shares per table, so a reader in another thread could release a writer's lock); every casacore lock taken with `nattempts=1` and checked with `haslock` (`lock()` fails silently); never wait for a lock. Other MAIN readers of the process (builds, lazy reads) are not in the mutex: MAIN keyword writes go through `partition_cache._write_main_keywords`, which takes the lock again before the change and the flush and retries on "should be locked".
  - Every read opens and closes its own table: no table of the MS is open after `open_datatree` returns or raises. Never xarray's `CachingFileManager`.
  - Every module-level memo or lock has an `os.register_at_fork(after_in_child=...)` reset (index, partition and pointing memos, write mutexes, the casatools lock, PR 1's sub-table cache state): add one for any new one. dask's default thread pool is not fork-safe: a fork child must use a new pool or another scheduler.
- **Tests:** `tests/unit/measurement_set/_utils/_msv2/`: `test_backend_*`, `test_msv2_backend_entrypoint.py`, `test_open_msv2_equivalence.py` (fast tier against the converter), `test_open_msv2_processes.py` (fork, processes, threads), `test_partition_cache*.py` (read/write, races and crash points, staleness against the converter as oracle), `test_open_msv2_corpus.py` (slow tier on downloaded MSs, marked `slow`, runs only with `XRADIO_SLOW_TESTS=1`); `tests/casatools/test_casatools_msv2_backend.py`; `tests/stakeholder/test_msv2_backend_stakeholder.py` (engine against the stakeholder conversions; the shared MSs only opened with `"off"`). Hygiene: the `_msv2` conftest sets `XRADIO_MSV2_PARTITION_CACHE=off` (autouse, function scope: module-scoped fixtures run before it, so pass `partition_cache="off"` there); the session MSs (`backend_ms`) are read-only (the session fails if a file of them changed): tests that store partitions or modify an MS use `ms_copy`; never open shared or downloaded MSs with `"auto"` (copy them first).

---

## Zarr v3 / Storage

- **`ZARR_FORMAT = 3`** — single source of truth at `src/xradio/_utils/zarr/config.py`. There is **no per-call override and no env var**; it is threaded as `zarr_format=ZARR_FORMAT` into every `to_zarr` / `zarr.open` call. Set it to `2` to revert all writes to Zarr v2 (e.g. for A/B perf comparison).
- **Default compressor / codec:** `zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="noshuffle")` (a v3 `BytesBytesCodec`), the default `compressor` of `convert_msv2_to_processing_set` and `_utils/_msv2/conversion.py::_convert_and_write_partition`. Chosen over the former `ZstdCodec(level=2)` because blosc-lz4 decompresses ~1.6× faster on high-entropy visibility data (faster loads) for a small ratio cost; `shuffle`/higher levels gave no benefit here (see `dev/LOAD_PERF_FINDINGS.md`). Encoding key is the **plural** `"compressors": (compressor,)` (v3), not v2's singular `"compressor"`.
- **Consolidated metadata:** not part of the v3 spec, but XRADIO still writes (`zarr.consolidate_metadata`) and reads it; `src/xradio/__init__.py` silences the resulting `ZarrUserWarning` and `UnstableSpecificationWarning`. The streamed write consolidates an MSv4 only after its last value: an MSv4 without consolidated metadata is incomplete (interrupted conversion), and xarray warns `Failed to open Zarr store with consolidated metadata` when opening it.
- `storage_backend="netcdf"` is **not implemented** (only `zarr` works).

### Performance debugging (this repo recently moved to Zarr v3; read/write got slow)

Most likely culprits, in order:
1. **Chunking** — `add_encoding()` (`_utils/_zarr/encoding.py`) defaults `chunks` to the **full dim sizes (one chunk per array)** when no chunksize is passed → kills parallel/compressed I/O. The converter's `main_chunksize=None` now gives time-only chunks of ~128 MiB (`default_main_chunksize`; `{}` = one chunk per variable), but `pointing_chunksize=None` and the other sub-xds are still one chunk per array. The image writer (`image/_util/_zarr/xds_to_zarr.py`) additionally enforces a 0.95 GiB max-chunk guard that raises `ValueError`.
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
conda run -n zinc python -m pytest                 # full suite (testpaths=[tests], --strict-markers, --import-mode=importlib)
conda run -n zinc python -m pytest tests/unit      # unit only
PATH=<env>/bin:$PATH PYTHONPATH=src make schema-export   # regenerate schemas/{VisibilityXds,SpectrumXds,ImageXds}.json
```

- `make schema-export` uses the `python` on `PATH`. Re-run it after any change to a schema class or schema docstring (image, MS and `schema/measures.py`): `tests/unit/schema/test_export.py` fails when `schemas/*.json` are out of date.

- **Public testing helpers** live in `xradio.testing` (pytest-free, reused by external projects / benchviper ASV benchmarks):
  - `assert_xarray_datasets_equal(test, true, *, rtol=1e-7, atol=0.0, check_attrs=True, check_encoding=False)`
  - `assert_attrs_dicts_equal(...)`
  - `xradio.testing.measurement_set`: `download_measurement_set(input_ms, directory="/tmp")`, `check_msv4_matches_descr`, `check_processing_set_matches_msv2_descr`
  - `xradio.testing.image`: `download_image`, `download_and_open_image`, `create_empty_test_image`, `assert_image_block_equal`, `remove_path`
- **Test data** downloads via `toolviper.utils.data.download` (default to `/tmp` for MS assets — avoids Dropbox table-locking issues).
- **Schema checking:** validate data with `check_dataset` / `check_array` / `check_dict` / `check_datatree`; call `.expect()` on the returned `SchemaIssues` to raise.
- **casatools tests** (`tests/casatools/`, see `tests/README.md`): the Linux and macOS workflows test with python-casacore only (the main backend); casatools-specific behaviour is tested with real casatools in `tests/casatools/`, whose modules skip only where casatools is not installed or python-casacore is importable, and error (not skip) if the shim cannot be imported (the casatools workflow: `pip uninstall python-casacore`, `casatools==6.7.0.31`, `--ignore=tests/unit/measurement_set`). The tests compare partitions, MAIN row reads (`read_rows`, grids, `check_partition_cells`) and full `convert_msv2_to_processing_set` outputs of downloaded test MSs with codec-independent fingerprints computed with python-casacore (`tests/casatools/reference_python_casacore.json`; regenerate with `python -m tests.casatools.make_reference` in a python-casacore env when the converter output changes on purpose or a dependency upgrade changes it; values from libm or interpolation, e.g. `OBSERVER_POSITION` and the interpolated ephemeris variables, are compared to 1e-12, everything else bit for bit), and count the getcol / getcell calls of the row reads on real casatools tables (`test_casatools_read_rows.py`). Known casatools differences the tests accept: Float/Complex cells come back as float64/complex128 (so do VISIBILITY/SPECTRUM/WEIGHT of the MSv4, same values), and the shim has no `partnames`, so `check_partition_cells` cannot verify storage and the read decides. Do not simulate casatools in the python-casacore suite.
- **CI**: reusable `nrao/gh-actions-templates-public` templates (linux + codecov, macos, casatools, integration, basic-schema-install, run-ipynb) plus `pre-commit.yml`, which runs every hook of `.pre-commit-config.yaml` on all files (ruff 0.12.5 check + format, pyupgrade, absolufy-imports, hygiene hooks). `cov_project="xradio"`, test path `tests/`.

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
requirement for XRADIO objects. XRADIO uses two cycle-free designs:

1. **Non-cached, strong accessors (`xr_img`, `xr_ms`; the default for new
   accessors).** Register the class with
   `xradio._utils.xarray_helpers.register_uncached_accessor(name, xr.Dataset | xr.DataTree)(Cls)`,
   never with `xr.register_dataset_accessor` / `register_datatree_accessor`.
   The descriptor builds a new accessor on every attribute access and never
   stores it on the object, so the accessor holds its object strongly
   (`self._xds` / `self._xdt`) without forming a cycle. Chained calls on
   temporaries (`ds.isel(...).xr_img.get_lm_cell_size()`,
   `open_datatree(...).xr_ms.sel(...)`) are fine. Because every access is a
   new instance, accessors must keep **no state between calls**. References:
   `image_xds.py` (`ImageXds`), `measurement_set_xdt.py` (`MeasurementSetXdt`).
2. **Cached, weak accessor (`xr_ps` only, because it caches `summary()`).**
   `processing_set_xdt.py` registers a module-level factory that builds the
   accessor and swaps its back-reference to a `weakref` (`_weaken()`); the
   object is reached only through the `_xdt` property, which raises
   `ReferenceError` once the tree is gone, and rebinding goes through a strong
   local (`xdt = self._xdt.assign(...); self._xdt = xdt; return xdt`). A
   temporary tree has nothing keeping it alive during the call: bind it to a
   local first (`xdt = f(...); xdt.xr_ps.summary()`).
3. **Never store accessor instances** in attributes, containers, or module
   state; use them inline (`ds.xr_x.method()`). A stored non-cached accessor
   keeps its object alive; a stored `xr_ps` raises `ReferenceError` once its
   tree is gone.
4. **Prove cycle-free death in a unit test** for every new accessor
   (`tests/unit/test_accessor_lifecycle.py`):
   ```python
   ds = make_dataset(); ds.xr_x  # touch the accessor
   wr = weakref.ref(ds); del ds
   assert wr() is None            # died by refcount, NO gc.collect() needed
   ```
5. **Never mutate `attrs["data_groups"]` or a group dict in place.** `isel`,
   `sel` and `load` return datasets that share the attrs dict itself;
   `copy(deep=False)`, `compute`, `where`, `chunk` and `drop_vars` share the
   nested group dicts. Build new mappings and assign them with
   `replace_data_groups`, `remove_variables_from_data_groups`,
   `delete_data_variables` or `remove_variable_references` from
   `xradio._utils.xarray_helpers` (the last two also drop the `flag` /
   `beam_fit_params` attributes that name a deleted variable).
6. **DataTree caveat (related, not accessor-specific):** `xarray.DataTree`
   parent↔child links are themselves strong reference cycles — any dropped
   tree is cyclic garbage. Consumers that drop task-owned trees must sever
   them (see `astroviper.utils.data_tree.release_data_tree`); XRADIO code
   should avoid keeping dropped subtrees reachable.

---

## Gotchas

- **Accessors require importing their SUBPACKAGE, not just `xradio`.** Bare `import xradio` registers nothing; `import xradio.measurement_set` registers `.xr_ps`/`.xr_ms` and `import xradio.image` registers `.xr_img` (registration is a side effect of the defining module's import). Without it they raise `AttributeError`. Calling a method on the wrong node type raises `InvalidAccessorLocation` (a `ValueError` subclass). Image zarr stores written by xradio <= 1.2.3 (for example AstroVIPER outputs and test truths) have `attrs["type"]="image"`; `open_image`/`load_image` upgrade them to `"image_dataset"` (and the other schema attributes) on read.
- **casacore is optional.** `convert_msv2_to_processing_set` / `estimate_conversion_memory_and_cores` are imported inside a `try/except`: if neither python-casacore nor casatools is importable they emit a `UserWarning` and are simply **absent** from the namespace. Image I/O imports the CASA readers and writer only for CASA images: FITS and zarr images open and write without them (keep FITS code astropy-only), and reading or writing a CASA image raises `ModuleNotFoundError` naming the packages.
- **MAIN data variables are placeholders during a streamed conversion.** Between `create_data_variables` and the write, `xds.VISIBILITY` etc. are one-chunk dask arrays that raise if computed: use only their shapes, dtypes and attributes there (`check_deferred_variables` checks every lazy variable has a deferred description). Never read MAIN with TaQL or whole-column `getcol` in the conversion: go through `_tables/read_rows.py` (its module docstring lists the python-casacore traps it avoids).
- **Opening an MSv2 with the `xradio_msv2` engine may write into it** (default `partition_cache="auto"`): the first open of a writable MS adds the `XRADIO_PARTITIONS` sub-table, a MAIN keyword and a HISTORY row. Use `"read"`/`"off"` (or `XRADIO_MSV2_PARTITION_CACHE`) for shared data, and copies in tests.
- **`use_table_iter` is a deprecated no-op** (`convert_msv2_to_processing_set`, `convert_and_write_partition`): `True` only emits a `DeprecationWarning`.
- **casacore table locking breaks on Dropbox-backed paths.** Run any casacore/CASA table operations (conversion, CASA-image read/write, related tests) in `$TMPDIR`, not in the Dropbox tree.
- **`storage_backend="netcdf"` is documented but NOT implemented.** Only `zarr` works.
- **No `[build-system]` table** in `pyproject.toml` (pip then builds with setuptools' legacy `__legacy__` backend); no `setup.py`/`setup.cfg`. The `[project.entry-points]` (xarray backends) work through that fallback: built wheels contain `entry_points.txt`, and setuptools' automatic src-layout discovery packages `src/xradio` and the top level module `src/_xradio_xarray_backends.py`. Add `[build-system]` before anything the legacy backend cannot express (dynamic version, explicit package discovery).
- **No `xradio.__version__`.** Version is hardcoded `version = "v1.2.4"` in `pyproject.toml` (note leading `v`; pip normalizes to `1.2.4`).
- **`persistence_mode` defaults to `"w-"`** (fail if `.ps.zarr` exists). Use `"a"` to append, `"w"` to overwrite.
- **Lint and format with ruff, through pre-commit** (`[tool.ruff]` in `pyproject.toml`, ruff pinned to 0.12.5 in `.pre-commit-config.yaml`; notebooks are linted too). CI runs every hook on all files. `make python-format` still calls Black and is not what CI checks.
- **`--strict-markers` is on** — `@pytest.mark.<custom>` without registration will error. The only custom marker, `slow`, is registered in `tests/unit/measurement_set/_utils/_msv2/conftest.py` (`pytest_configure`, not `pyproject.toml`).
- **Two divergent docs dep lists**: the pyproject `docs` extra vs `docs/sphinx.txt` (the latter, pinned `Sphinx==7.2.6` etc., is what RTD actually installs — and RTD installs the `casacore` extra, not `docs`).
- **Notebooks under `docs/source/**/guides` and tutorials are executed by `run-ipynb` CI** — breaking the public API can fail notebook CI even when pytest passes.
- **Image datasets have a formal schema** (`xradio.image.schema.ImageXds`, `check_image`), but no I/O path calls the checker: validation is opt-in.
- **FITS:** `open_image` and the `xradio_fits_image` engine read FITS and `write_image(..., out_format="fits")` writes it; `load_image` does not read FITS. Compressed (`CompImageHDU`) or scaled (`BSCALE != 1.0` / `BZERO != 0.0`) FITS are incompatible with memmap and raise `RuntimeError`.
- **`open_processing_set` ≠ `read_image`**; the image reader is `open_image`. No `read_image` exists anywhere.
- **Stray `.DS_Store` files** are committed throughout the tree — ignore them.
- **Two different "schema" things:** the typed dataclass framework in `src/xradio/schema/` vs the unrelated runtime mapper `src/xradio/_utils/schema.py` (`convert_generic_xds_to_xradio_schema`). Don't confuse them.
- **Decorated schema classes are not normal objects** — `@xarray_*_schema` overrides `__new__`, so `MySchema(...)` returns a validated `xr.DataArray`/`Dataset`/`dict`; `isinstance(obj, MySchema)` is `False`. Use `check_dataset` / `check_array`.
- **`xradio_logger()` returns a MODULE** (toolviper's logger module, else stdlib `logging`), not a `Logger` instance — call `.debug(...)` / `.info(...)` directly on it.
