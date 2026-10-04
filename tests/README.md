# XRADIO Tests
XRADIO Tests use pytest and are located in the `tests` directory. There are three types of tests:
- Unit Tests: located in `tests/unit`
- stakeholder Tests: located in `tests/stakeholder`
- casatools Tests: located in `tests/casatools`. They run only where xradio reads MSv2 with
  casatools, i.e. casatools is installed and python-casacore is not (the casatools test
  workflow, `.github/workflows/python-testing-casatools.yml`); everywhere else (the Linux and
  macOS workflows, which use python-casacore) they are skipped. They compare the partitions,
  MAIN-table reads and conversions of downloaded test MSs with reference values computed with
  python-casacore (`tests/casatools/reference_python_casacore.json`). Regenerate that file, in
  an environment with python-casacore, with `python -m tests.casatools.make_reference` (from
  the repository root) when the converter output changes on purpose.

The main documentation for tests is in [xradio.testing](https://github.com/casangi/xradio/src/xradio/testing/README.md).
