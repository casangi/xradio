# AGENT.md — XRADIO

Guide for AI coding agents (and humans) working in this repository. Repo root: `/Users/jsteeb/Dropbox/viper_dev/xradio`.

## What is XRADIO

XRADIO (**X**array **Radio** Astronomy **D**ata **IO**) is a pure-Python library that defines and implements xarray-based data schemas for radio-astronomy data. Its production schema is the **Measurement Set v4 (MS v4)**, surfaced on disk as a single `.ps.zarr` **Processing Set** — a 3-level `xarray.DataTree` (Processing Set → MS v4 nodes → metadata sub-datasets) backed by **Zarr v3**. It provides backends to convert CASA Measurement Set v2 (casacore tables) → MS v4, lazily open / eagerly load Processing Sets, read/write **Image** cubes (CASA / FITS / Zarr), and validate data against typed schemas. There is deliberately **no direct MSv2 reader** — you convert once. XRADIO is part of the casangi / VIPER stack (depends on `toolviper` for logging, Dask clients, test-data download, and memory management).

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
│    entry points, imports only os + xarray.backends, so loading them never imports xradio)
│   ├── measurement_set/      # PRIMARY subsystem: PS / MS v4 IO + MSv2->MSv4 conversion
│   │   ├── __init__.py        #   public API (convert wrapped in try/except for missing casacore)
│   │   ├── convert_msv2_to_processing_set.py
│   │   ├── open_processing_set.py / load_processing_set.py
│   │   ├── processing_set_xdt.py    # .xr_ps accessor (ProcessingSetXdt)
│   │   ├── measurement_set_xdt.py   # .xr_ms accessor (MeasurementSetXdt)
│   │   ├── schema.py                # VisibilityXds, SpectrumXds, sub-dataset & info schemas
│   │   └── _utils/_msv2|_zarr|_utils # internal converters, casacore readers, zarr encoding
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
│   └── stakeholder/          # higher-level integration tests
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
where `ms_v4_id` is the **zero-padded** partition index (`conversion.py:1392`; `convert_msv2_to_processing_set.py:197`). Embedding the input-MS basename is what lets you append several distinct MSs into one PS without overwriting. ⚠ `.replace(".ms", "")` strips **every** literal `.ms` substring, not just the suffix.

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

- **Partitioning** (`_utils/_msv2/partition_queries.py::create_partitions`): reads MAIN + FIELD/SOURCE/STATE subtables once into a pandas DataFrame and groupby-aggregates into a list of partition dicts. **Mandatory split keys** are always prepended (`partition_queries.py:46-51`): `[DATA_DESC_ID, OBS_MODE, OBSERVATION_ID, EPHEMERIS_ID]` → data description (= spectral window + polarization setup) and observation mode are always split. User keys appended: `FIELD_ID, SCAN_NUMBER, STATE_ID, SOURCE_ID, SUB_SCAN_NUMBER, ANTENNA1`. Keys absent from the data are dropped before groupby.
- **`parallel_mode` dispatch** (`convert_msv2_to_processing_set.py:198-248`): `partition` wraps each `convert_and_write_partition` in `dask.delayed`, then `dask.compute(delayed_list)`. `time` hard-asserts `len(partitions)==1` (`:164-167`).
- **Per-partition conversion** (`_utils/_msv2/conversion.py::convert_and_write_partition`, `l.1001`): builds a TaQL `WHERE` (`create_taql_query_where`, `l.764`); reshapes flat MSv2 rows into the dense `(time, baseline_id, frequency, polarization)` grid via `tidxs`/`bidxs` (`calc_indx_for_row_split`, `l.390`); assembles `ms_xdt.ds` + sub-xds child nodes (`l.1393-1416`); applies chunking + `add_encoding`; writes via `to_zarr(..., zarr_format=ZARR_FORMAT)` to `out_file/<ms_v4_name>` (`l.1420-1424`).
- **Finalize** (`convert_msv2_to_processing_set.py`): reopens root, sets `attrs["type"]="processing_set"` (`:254`), calls `zarr.consolidate_metadata(...)` (`:255`).
- **casacore read path**: table-touching modules do `from casacore import tables` and fall back to `xradio._utils._casacore.casacore_from_casatools` (a casatools shim with python-casacore's API) when python-casacore is missing. CASA images are read through `casacore.images` (or the shim); FITS images only through astropy. Tables open read-only (`lockoptions={"option":"usernoread"}`, `ack=False`). `SOURCE_MODEL` (frequently corrupted) is never read. The MS test-data generator (`xradio.testing.measurement_set.msv2_io`) builds MSv2 files with the same table backends.

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
- **casacore table locking breaks on Dropbox-backed paths.** Run any casacore/CASA table operations (conversion, CASA-image read/write, related tests) in `$TMPDIR`, not in the Dropbox tree.
- **`storage_backend="netcdf"` is documented but NOT implemented.** Only `zarr` works.
- **No `[build-system]` table** in `pyproject.toml` (pip then builds with setuptools' legacy `__legacy__` backend); no `setup.py`/`setup.cfg`. The `[project.entry-points]` (xarray backends) work through that fallback: built wheels contain `entry_points.txt`, and setuptools' automatic src-layout discovery packages `src/xradio` and the top level module `src/_xradio_xarray_backends.py`. Add `[build-system]` before anything the legacy backend cannot express (dynamic version, explicit package discovery).
- **No `xradio.__version__`.** Version is hardcoded `version = "v1.2.4"` in `pyproject.toml` (note leading `v`; pip normalizes to `1.2.4`).
- **`persistence_mode` defaults to `"w-"`** (fail if `.ps.zarr` exists). Use `"a"` to append, `"w"` to overwrite.
- **Lint and format with ruff, through pre-commit** (`[tool.ruff]` in `pyproject.toml`, ruff pinned to 0.12.5 in `.pre-commit-config.yaml`; notebooks are linted too). CI runs every hook on all files. `make python-format` still calls Black and is not what CI checks.
- **`--strict-markers` is on but no custom markers are registered** — `@pytest.mark.<custom>` without registration will error.
- **Two divergent docs dep lists**: the pyproject `docs` extra vs `docs/sphinx.txt` (the latter, pinned `Sphinx==7.2.6` etc., is what RTD actually installs — and RTD installs the `casacore` extra, not `docs`).
- **Notebooks under `docs/source/**/guides` and tutorials are executed by `run-ipynb` CI** — breaking the public API can fail notebook CI even when pytest passes.
- **Image datasets have a formal schema** (`xradio.image.schema.ImageXds`, `check_image`), but no I/O path calls the checker: validation is opt-in.
- **FITS:** `open_image` and the `xradio_fits_image` engine read FITS and `write_image(..., out_format="fits")` writes it; `load_image` does not read FITS. Compressed (`CompImageHDU`) or scaled (`BSCALE != 1.0` / `BZERO != 0.0`) FITS are incompatible with memmap and raise `RuntimeError`.
- **`open_processing_set` ≠ `read_image`**; the image reader is `open_image`. No `read_image` exists anywhere.
- **Stray `.DS_Store` files** are committed throughout the tree — ignore them.
- **Two different "schema" things:** the typed dataclass framework in `src/xradio/schema/` vs the unrelated runtime mapper `src/xradio/_utils/schema.py` (`convert_generic_xds_to_xradio_schema`). Don't confuse them.
- **Decorated schema classes are not normal objects** — `@xarray_*_schema` overrides `__new__`, so `MySchema(...)` returns a validated `xr.DataArray`/`Dataset`/`dict`; `isinstance(obj, MySchema)` is `False`. Use `check_dataset` / `check_array`.
- **`xradio_logger()` returns a MODULE** (toolviper's logger module, else stdlib `logging`), not a `Logger` instance — call `.debug(...)` / `.info(...)` directly on it.
