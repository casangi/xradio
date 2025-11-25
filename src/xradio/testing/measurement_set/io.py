"""
I/O helpers to create derivative datasets (processing sets, MSv4 partitions) and
download reusable MeasurementSets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Any

from toolviper.utils.data import download

from xradio.measurement_set import convert_msv2_to_processing_set
from xradio.measurement_set._utils._msv2.conversion import convert_and_write_partition


def download_measurement_set(input_ms: str, directory: str | Path = "/tmp") -> Path:
    """
    Download a MeasurementSet v2 archive into the given directory.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    download(file=input_ms, folder=str(directory))
    return directory / input_ms


def build_processing_set_from_msv2(
    
    in_file: str | Path,
    out_file: str | Path,
    *,
    partition_scheme: Optional[Iterable[dict]] = None,
    overwrite: bool = False,
    parallel_mode: str = "partition",
    **convert_kwargs: Any,
) -> Path:
    """
    Convert an MSv2 dataset into a processing set using the production converter.
    """

    convert_msv2_to_processing_set(
        in_file=str(in_file),
        out_file=str(out_file),
        partition_scheme=list(partition_scheme or []),
        overwrite=overwrite,
        parallel_mode=parallel_mode,
        **convert_kwargs,
    )
    return Path(out_file)


def build_msv4_partition(
    ms_path: str | Path,
    out_root: str | Path,
    *,
    msv4_id: str = "msv4id",
    partition_kwargs: Optional[Dict[str, Any]] = None,
    use_table_iter: bool = False,
    overwrite: bool = True,
) -> Path:
    """
    Convert a MeasurementSet v2 partition into an MSv4 Zarr tree.
    """

    partition_kwargs = partition_kwargs or {"DATA_DESC_ID": [0]}
    convert_and_write_partition(
        str(ms_path),
        str(out_root),
        msv4_id,
        partition_kwargs,
        use_table_iter=use_table_iter,
        overwrite=overwrite,
    )
    return Path(out_root) / f"{Path(ms_path).stem}_{msv4_id}"


def build_minimal_msv4_xdt(
    ms_path: str | Path,
    *,
    out_root: str | Path | None = None,
    msv4_id: str = "msv4id",
    partition_kwargs: Optional[Dict[str, Any]] = None,
    use_table_iter: bool = False,
    overwrite: bool = True,
) -> Path:
    """
    Convenience wrapper that selects reasonable defaults for the minimal MSv4 conversion.
    """

    if out_root is None:
        out_root = Path(f"{Path(ms_path).stem}_processing_set.zarr")
    return build_msv4_partition(
        ms_path,
        out_root,
        msv4_id=msv4_id,
        partition_kwargs=partition_kwargs,
        use_table_iter=use_table_iter,
        overwrite=overwrite,
    )

