"""
I/O helpers to download reusable MeasurementSets.

Casacore-dependent functions have been moved to msv2_io.py.
"""

from __future__ import annotations

from pathlib import Path

from toolviper.utils.data import download


def download_measurement_set(input_ms: str, directory: str | Path = "/tmp") -> Path:
    """
    Download a MeasurementSet v2 archive into the given directory.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    download(file=input_ms, folder=str(directory))
    return directory / input_ms
