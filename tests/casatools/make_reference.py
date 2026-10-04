"""
Write the reference values of the casatools tests (reference_python_casacore.json)
with python-casacore, the main backend of xradio: the fingerprints of the
partitions, MAIN row reads and conversions of the cases in reference.py.

Run it from the repository root, in an environment with python-casacore and
this xradio installed (or on PYTHONPATH):

    python -m tests.casatools.make_reference [--ms-dir DIR] [--out-dir DIR]

The test MSs are downloaded into --ms-dir if they are not there yet. Every
conversion case is converted with each of its variants: all must give the same
fingerprint (the variants change only chunks, batches and parallel_mode).
Regenerate the file when the converter output changes on purpose (e.g. a new
variable or attribute); the casatools tests then compare against it.
"""

import argparse
import json
import pathlib
import shutil
import sys
import tempfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ms-dir", default=None, help="Test MS folder")
    parser.add_argument(
        "--out-dir", default=None, help="Folder for the (deleted) conversions"
    )
    parser.add_argument(
        "--output", default=None, help="Reference file (default: the tests' file)"
    )
    args = parser.parse_args(argv)

    # python-casacore must be the backend (it is whenever it is installed)
    import casacore
    import numpy as np

    from tests.casatools import reference as ref
    from xradio.measurement_set._utils._msv2._tables.read_rows import (
        backend_has_in_place_reads,
    )

    if not backend_has_in_place_reads():
        raise RuntimeError("xradio does not read MSv2 with python-casacore here")
    ms_dir = pathlib.Path(args.ms_dir) if args.ms_dir else ref.MS_DIR
    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="casatools_ref_", dir=args.out_dir))
    output = pathlib.Path(args.output) if args.output else ref.REFERENCE_FILE

    result = {
        "generated_with": {
            "python-casacore": casacore.__version__,
            "numpy": np.__version__,
        },
        "reads": {},
        "conversions": {},
        "nodes": {},
    }
    try:
        for name, case in ref.READ_CASES.items():
            print(f"reads {name}", flush=True)
            result["reads"][name] = ref.main_table_fingerprint(
                ref.ms_path(case["ms"], ms_dir),
                case["partition_scheme"],
                case["read_columns"],
            )
        for case, variants in ref.CASE_VARIANTS.items():
            fingerprints, attempts = {}, {}
            for variant in variants:
                print(f"conversion {case} {variant}", flush=True)
                with ref.ConversionSpy() as spy:
                    ps_path = ref.convert(
                        case, variant, out_dir / f"{case}_{variant}.ps.zarr", ms_dir
                    )
                fingerprints[variant] = ref.processing_set_fingerprint(ps_path)
                attempts[variant] = spy.attempts
                shutil.rmtree(ps_path)
            default = fingerprints[variants[0]]
            for variant in variants[1:]:
                diffs = ref.fingerprint_differences(default, fingerprints[variant])
                if attempts[variant] != attempts[variants[0]]:
                    diffs.append(f"attempts {attempts[variant]}")
                if diffs:
                    raise RuntimeError(
                        f"{case}: variant {variant} differs from {variants[0]}: "
                        + "; ".join(diffs[:10])
                    )
            result["nodes"].update(default["nodes"])
            result["conversions"][case] = {
                "msv4": default["msv4"],
                "attrs": default["attrs"],
                "attempts": attempts[variants[0]],
            }
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    with open(output, "w") as out:
        json.dump(result, out, indent=0, sort_keys=True)
        out.write("\n")
    print(f"wrote {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
