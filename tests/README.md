# XRADIO Tests
XRADIO Tests use pytest and are located in the `tests` directory. There are three types of tests:
- Unit Tests: located in `tests/unit`
- stakeholder Tests: located in `tests/stakeholder`
- casatools Tests: located in `tests/casatools` (see [below](#casatools-tests))

The main documentation for tests is in [xradio.testing](https://github.com/casangi/xradio/src/xradio/testing/README.md).

## casatools tests

xradio reads MSv2 with python-casacore when it is importable, otherwise with casatools, through
the shim `xradio._utils._casacore.casacore_from_casatools`. The Linux and macOS workflows test
with python-casacore only, and the python-casacore tests do not simulate casatools: what is
specific to casatools is tested with real casatools in `tests/casatools`.

**When they run.** Only in the casatools workflow (`.github/workflows/python-testing-casatools.yml`:
python-casacore uninstalled, casatools installed, on Linux and macOS). Every module calls
`reference.skip_unless_casatools_backend()`, which skips it only where casatools is not installed
(`importlib.util.find_spec("casatools") is None`) or python-casacore is importable (the Linux and
macOS workflows, which collect and skip them). Otherwise it imports the shim, and if that fails
the module errors, it is not skipped: a casatools install that cannot be imported fails the
casatools workflow.

**What they test**, on test MSs downloaded into `/tmp/test` (as the stakeholder tests do):
- `test_casatools_backend.py`: xradio uses the shim, without in-place reads or the sub-table cache;
- `test_casatools_main_reads.py`: the partitions, their MAIN rows, the row reads of every MAIN
  column, the converter's grids and `check_partition_cells`, compared with the reference values
  (below);
- `test_casatools_conversion.py`: `convert_msv2_to_processing_set` of several MSs and options,
  compared with the reference values;
- `test_casatools_read_rows.py`: the getcol reads of `read_rows`, `read_row_range`,
  `read_column_rows`, `read_rows_to_grid` and the shape scan of `check_partition_cells`: the
  values (against cells read with getcell) and the calls made (counted by wrapping the methods
  of the opened casatools table), as the python-casacore unit tests check the in-place reads.

Not covered by real-casatools tests: no downloadable test MS has a MAIN column with both
defined and undefined cells (in all of them every array column of the MAIN table has all its
cells defined, or none), so there is no casatools test of a read across undefined cells between
defined ones, of a bridged gap over undefined cells (`read_rows_to_grid` falls back to the
partition rows: tested with a gap of cells of another shape), or of the shape scan of
`check_partition_cells` finding an undefined cell after a defined first cell (tested with a cell
of another shape). The python-casacore unit tests cover these cases for the in-place reads.

### The reference values

`test_casatools_main_reads.py` and `test_casatools_conversion.py` compare codec-independent
fingerprints (digests of the partitions, of the row reads, and of every variable, coordinate and
attribute of the processing sets) with those that python-casacore gives, stored in
`tests/casatools/reference_python_casacore.json`. Only the casatools workflow checks this file:
the Linux and macOS workflows skip these tests, so a change that makes it stale fails only the
casatools workflow.

The values are compared bit for bit, except:
- casatools returns Float, Complex and Int cells as float64, complex128 and int64: the data
  variables of the MSv4s and the grids read from such columns are wider (same values);
- the values computed with libm or by interpolation (`OBSERVER_POSITION`, and the ephemeris
  variables interpolated to the MSv4 times, such as `FIELD_PHASE_CENTER_DIRECTION` of the ALMA
  MS: see `reference.approx_variables`), whose last bits may differ on macOS arm64 (the file is
  computed on Linux x86-64), are kept as float values in the file and compared to 1e-12 of the
  largest magnitude of the variable; their dtype, dims, shape and attributes are compared bit
  for bit.

**When to regenerate it:**
- the converter output changes on purpose: other partitions, MSv4 names, variables,
  coordinates, values or attributes;
- a dependency upgrade changes the output, e.g. attributes that xarray, zarr, astropy or the
  schema write differently, or values that a new version computes differently;
- the cases or the fingerprints of `tests/casatools/reference.py` change.

Do not regenerate it to make a difference between casatools and python-casacore go away: those
differences are what these tests are for.

**How:** in an environment with python-casacore (xradio then reads MSv2 with it) and this xradio
installed or on `PYTHONPATH`, from the repository root:

```sh
python -m tests.casatools.make_reference [--ms-dir DIR] [--out-dir DIR] [--output FILE]
```

It downloads the test MSs into `/tmp/test` (or `--ms-dir`) if they are not there, computes the
reads, converts every case with each of its variants (which must all give the same
fingerprint), and writes the file (in about a minute). Regenerating gives the same file byte for
byte: check that a second run (`--output` to another file) is identical, review the diff, and
commit the file with the change that caused it. Then run `tests/casatools` with casatools (as
the casatools workflow does) to check it.
