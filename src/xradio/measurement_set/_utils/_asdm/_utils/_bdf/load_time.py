"""
Loads the times (time centers, durations, actual times and actual durations) of the
integrations stored in the BDFs of a partition.

Times are produced per integration: one per subset for the usual BDFs
(dimensionality 1, one integration per subset), and numTime per subset for packed
BDFs (dimensionality 0, all the integrations (TIM samples) in one subset). Absolute
times are float64 seconds since the Unix epoch (see
:func:`xradio.measurement_set._utils._asdm._utils.time.convert_time_asdm_to_unix`),
durations are seconds.

The times of a BDF are the same for all the SPWs/partitions that use it, so they are
cached per BDF file (keyed by real path, modification time and size).
"""

import os
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyasdm

from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils._bdf import config
from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
    open_bdf,
)
from xradio.measurement_set._utils._asdm._utils.time import convert_time_asdm_to_unix

#: Name of the CSV file where BDF header information is saved when
#: config.do_save_blob_info is enabled (debugging aid)
BLOB_INFO_CSV_PATH = "xradio_asdm_blob_header_info_etc.csv"

#: Maximum number of BDFs whose times are kept in the cache
BDF_TIMES_CACHE_MAX_ENTRIES = 4096

_bdf_times_cache: OrderedDict = OrderedDict()
_bdf_times_cache_lock = threading.Lock()


def load_times_from_partition_bdfs(
    bdf_paths: list[str], scans_metadata: pd.DataFrame | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load the time information of the integrations of a partition/MSv4 from its BDFs.

    Parameters
    ----------
    bdf_paths : list[str]
        Paths to the BDF (Binary Data Format) files of the partition, in time order.
    scans_metadata : pd.DataFrame | None
        Unused. Kept for backward compatibility of the signature (there is no
        fallback to the Scan table times: errors reading the BDF times are raised).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]
        - time_centers: midpoint of every integration (float64 seconds since the
          Unix epoch)
        - durations: nominal duration of every integration (seconds)
        - actual_times: actual (measured) time of every integration (seconds since
          the Unix epoch), the midpoint when not available
        - actual_durations: actual (measured) duration of every integration
          (seconds), the nominal duration when not available
        - time_indices_by_bdf: dict with the "bdf_names" (list of BDF paths) and
          "bdf_start" (list with the index of the first integration of every BDF,
          plus the total number of integrations as last element)

    Raises
    ------
    ValueError
        If no BDF paths are given.
    RuntimeError
        If a BDF cannot be opened or its header parsed (BDFOpenError), or its
        times cannot be read. The message names the BDF.
    """
    return load_times_from_bdfs(bdf_paths)


def make_blob_info(bdf_header: pyasdm.bdf.BDFHeader) -> pd.DataFrame:
    """
    Make a one-row DataFrame with information from a BDF header (debugging aid).

    Parameters
    ----------
    bdf_header : pyasdm.bdf.BDFHeader
        Header of a BDF.

    Returns
    -------
    pd.DataFrame
        One row indexed by (execblock_uid, dataOID).
    """
    basebands_info = ""
    for baseband in bdf_header.getBasebandsList():
        basebands_info += f"{baseband['name']} "
        for spw in baseband["spectralWindows"]:
            spw_idx = f"spw_{spw['sw']}"
            spectral_points = spw["numSpectralPoint"]
            num_bin = spw["numBin"]
            cross_pol_products = len(spw["crossPolProducts"])
            sd_pol_products = len(spw["sdPolProducts"])
            basebands_info += f"{spw_idx} {spectral_points} {num_bin} {cross_pol_products} {sd_pol_products}"

    bdf_info = {
        "execblock_uid": bdf_header.getExecBlockUID(),
        "dataOID": bdf_header.getDataOID(),
        "title": bdf_header.getTitle(),
        "correlation_mode": bdf_header.getCorrelationMode(),
        "processor_type": bdf_header.getProcessorType(),
        "spectral_resolution_type": bdf_header.getSpectralResolutionType(),
        "apc_list": " ".join(map(str, bdf_header.getAPClist())),
        "dimensionality": bdf_header.getDimensionality(),
        "num_time": bdf_header.getNumTime(),
        "binary_types": " ".join(map(str, bdf_header.getBinaryTypes())),
        "num_antenna": bdf_header.getNumAntenna(),
        "actual_times_size": bdf_header.getSize("actualTimes"),
        "actual_times_axes": " ".join(map(str, bdf_header.getAxes("actualTimes"))),
        "actual_durations_size": bdf_header.getSize("actualDurations"),
        "actual_durations_axes": " ".join(
            map(str, bdf_header.getAxes("actualDurations"))
        ),
        "flags_size": bdf_header.getSize("flags"),
        "flags_axes": " ".join(map(str, bdf_header.getAxes("flags"))),
        "auto_data_size": bdf_header.getSize("autoData"),
        "auto_data_axes": " ".join(map(str, bdf_header.getAxes("autoData"))),
        "cross_data_size": bdf_header.getSize("crossData"),
        "cross_data_axes": " ".join(map(str, bdf_header.getAxes("crossData"))),
        "zero_lags_size": bdf_header.getSize("zeroLags"),
        "zero_lags_axes": " ".join(map(str, bdf_header.getAxes("zeroLags"))),
        "basebands_spws_points_bins_crossx_sdx": basebands_info,
    }
    blob_info = pd.DataFrame([bdf_info]).set_index(["execblock_uid", "dataOID"])

    return blob_info


def save_blob_info(csv_path: str, blob_info: pd.DataFrame) -> None:
    """
    Append BDF header information to a CSV file (writing the header row only if
    the file is new or empty).

    Parameters
    ----------
    csv_path : str
        Path of the CSV file.
    blob_info : pd.DataFrame
        Information produced by :func:`make_blob_info`.
    """
    do_header = not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0)
    blob_info.to_csv(csv_path, mode="a", header=do_header)


def load_times_from_bdfs(
    bdf_paths: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Read the timing information of every integration from a list of BDFs.

    Every BDF is read once (and cached, see :func:`load_times_bdf_cached`).

    Parameters
    ----------
    bdf_paths : list[str]
        List of paths to BDF files to be processed, in time order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]
        - time_centers : midpoint of every integration (seconds since the Unix epoch)
        - durations : nominal duration of every integration (seconds)
        - actual_times : actual time of every integration (seconds since the Unix
          epoch), the midpoint when not available
        - actual_durations : actual duration of every integration (seconds), the
          nominal duration when not available
        - time_indices_by_bdf: dict with "bdf_names" (BDF paths) and "bdf_start"
          (cumulative index of the first integration of every BDF, with
          len(bdf_names) + 1 elements)

    Raises
    ------
    ValueError
        If bdf_paths is empty.
    RuntimeError
        If a BDF cannot be opened or its header parsed (BDFOpenError), or its
        times cannot be read. The message names the BDF.
    """
    bdf_names = [str(bdf_path) for bdf_path in bdf_paths]
    if not bdf_names:
        raise ValueError("Expected at least one BDF path to load times from")

    all_times = [load_times_bdf_cached(bdf_path) for bdf_path in bdf_names]

    bdf_start = [0]
    for time_centers, _, _, _ in all_times:
        bdf_start.append(bdf_start[-1] + len(time_centers))
    time_indices_by_bdf = {"bdf_names": bdf_names, "bdf_start": bdf_start}

    time_centers, durations, actual_times, actual_durations = (
        np.concatenate([times[idx] for times in all_times]) for idx in range(4)
    )

    return (
        time_centers,
        durations,
        actual_times,
        actual_durations,
        time_indices_by_bdf,
    )


def _bdf_times_cache_key(bdf_path: str) -> tuple | None:
    try:
        stat = os.stat(bdf_path)
    except OSError:
        return None
    return (os.path.realpath(bdf_path), stat.st_mtime_ns, stat.st_size)


def clear_bdf_times_cache() -> None:
    """Empty the cache of BDF times used by :func:`load_times_bdf_cached`."""
    with _bdf_times_cache_lock:
        _bdf_times_cache.clear()


def load_times_bdf_cached(
    bdf_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Same as :func:`load_times_bdf`, caching the results per BDF file.

    The cache key is (real path, modification time, size) of the file, so a
    modified file is read again. Paths that cannot be stat'ed are not cached.
    The arrays returned are copies (callers can modify them).

    Parameters
    ----------
    bdf_path : str
        Path to a BDF.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        See :func:`load_times_bdf`.
    """
    key = _bdf_times_cache_key(bdf_path)
    if key is not None:
        with _bdf_times_cache_lock:
            cached = _bdf_times_cache.get(key)
            if cached is not None:
                _bdf_times_cache.move_to_end(key)
        if cached is not None:
            return tuple(times.copy() for times in cached)

    times = load_times_bdf(bdf_path)

    if key is not None:
        with _bdf_times_cache_lock:
            _bdf_times_cache[key] = tuple(times_var.copy() for times_var in times)
            while len(_bdf_times_cache) > BDF_TIMES_CACHE_MAX_ENTRIES:
                _bdf_times_cache.popitem(last=False)

    return times


def _times_per_subset(bdf_header: pyasdm.bdf.BDFHeader, bdf_path: str) -> int:
    """
    Number of integrations (TIM samples) per subset: numTime for packed BDFs
    (dimensionality 0), 1 otherwise.
    """
    if bdf_header.getDimensionality() == 0:
        num_time = int(bdf_header.getNumTime())
        if num_time < 1:
            raise RuntimeError(
                f"Packed BDF (dimensionality 0) with invalid numTime={num_time}: "
                f"{bdf_path}"
            )
        return num_time

    return 1


def _split_subset_times(
    midpoint_ns: int, interval_ns: int, num_tim: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Midpoints (integer ns, ASDM ArrayTime) and durations (ns) of the num_tim
    integrations of a subset, splitting the subset interval into num_tim equal
    parts: t_i = (midpoint - interval/2) + (i + 0.5) * interval / num_tim.
    """
    midpoint_ns = int(midpoint_ns)
    interval_ns = int(interval_ns)
    if num_tim == 1:
        return np.array([midpoint_ns], dtype=np.int64), np.array(
            [interval_ns], dtype=np.float64
        )

    tim_idx = np.arange(num_tim, dtype=np.int64)
    tim_midpoints = midpoint_ns + ((2 * tim_idx + 1 - num_tim) * interval_ns) // (
        2 * num_tim
    )
    tim_durations = np.full(num_tim, interval_ns / num_tim, dtype=np.float64)
    return tim_midpoints, tim_durations


def _per_tim_values(component: dict | None, num_tim: int) -> np.ndarray | None:
    """
    One value per integration (TIM sample) from an actualTimes/actualDurations
    binary component (the first value of every TIM sample, other axes such as ANT
    are ignored). None if the component is not present or does not have
    num_tim * N values.
    """
    if not component or not component.get("present"):
        return None
    arr = component.get("arr")
    if arr is None:
        return None
    arr = np.asarray(arr).ravel()
    if arr.size == 0 or arr.size % num_tim != 0:
        return None
    return arr.reshape(num_tim, -1)[:, 0]


def _actual_or_nominal(
    actual_times_ns: np.ndarray | None,
    actual_durations_ns: np.ndarray | None,
    tim_midpoints_ns: np.ndarray,
    tim_durations_ns: np.ndarray,
    bdf_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Actual times (integer ns) and durations (ns), falling back to the nominal ones
    when not available or not plausible (actual time more than one integration
    duration away from the midpoint, actual duration not in (0, 2 * duration]).
    """
    times_ns = tim_midpoints_ns
    if actual_times_ns is not None:
        actual_times_ns = actual_times_ns.astype(np.int64)
        valid = np.abs(actual_times_ns - tim_midpoints_ns) <= np.maximum(
            tim_durations_ns, 1
        )
        if not np.all(valid):
            xradio_logger().warning(
                f"Implausible actualTimes in BDF {bdf_path} ({np.sum(~valid)} values "
                "far from the integration midpoints). Using the midpoints instead."
            )
        times_ns = np.where(valid, actual_times_ns, tim_midpoints_ns)

    durations_ns = tim_durations_ns
    if actual_durations_ns is not None:
        actual_durations_ns = actual_durations_ns.astype(np.float64)
        valid = (actual_durations_ns > 0) & (
            actual_durations_ns <= 2 * tim_durations_ns
        )
        if not np.all(valid):
            xradio_logger().warning(
                f"Implausible actualDurations in BDF {bdf_path} ({np.sum(~valid)} "
                "values). Using the nominal durations instead."
            )
        durations_ns = np.where(valid, actual_durations_ns, tim_durations_ns)

    return times_ns, durations_ns


def load_times_bdf(
    bdf_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the times of all the integrations of one BDF (opening the file once).

    Parameters
    ----------
    bdf_path : str
        Path to a BDF.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        float64 arrays with one element per integration (one per subset, or
        numTime per subset for packed BDFs):

        - time centers (midpoints), seconds since the Unix epoch
        - nominal durations, seconds
        - actual times, seconds since the Unix epoch (midpoints when the
          actualTimes binary component is absent or implausible)
        - actual durations, seconds (nominal durations when the actualDurations
          binary component is absent or implausible)

    Raises
    ------
    RuntimeError
        If the BDF cannot be opened or its header parsed (BDFOpenError, a
        RuntimeError), or its subsets cannot be read. The message names the BDF.

    Notes
    -----
    For packed BDFs (dimensionality 0, numTime=N integrations in one subset) the
    subset interval is split into N equal parts: integration i is centered at
    ``(midpoint - interval / 2) + (i + 0.5) * interval / N`` and lasts
    ``interval / N``, unless per-integration actual times/durations are present.
    """
    with open_bdf(bdf_path) as bdf_reader:
        bdf_header = bdf_reader.getHeader()
        if config.do_save_blob_info:
            save_blob_info(BLOB_INFO_CSV_PATH, make_blob_info(bdf_header))

        num_tim = _times_per_subset(bdf_header, bdf_path)

        midpoints_ns, durations_ns, actual_times_ns, actual_durations_ns = (
            [],
            [],
            [],
            [],
        )
        subset_idx = 0
        try:
            while bdf_reader.hasSubset():
                subset = bdf_reader.getSubset(
                    loadOnlyComponents={"actualTimes", "actualDurations"}
                )
                tim_midpoints, tim_durations = _split_subset_times(
                    subset["midpointInNanoSeconds"],
                    subset["intervalInNanoSeconds"],
                    num_tim,
                )
                tim_actual_times, tim_actual_durations = _actual_or_nominal(
                    _per_tim_values(subset.get("actualTimes"), num_tim),
                    _per_tim_values(subset.get("actualDurations"), num_tim),
                    tim_midpoints,
                    tim_durations,
                    bdf_path,
                )
                midpoints_ns.append(tim_midpoints)
                durations_ns.append(tim_durations)
                actual_times_ns.append(tim_actual_times)
                actual_durations_ns.append(tim_actual_durations)
                subset_idx += 1
        except Exception as exc:
            raise RuntimeError(
                f"Error while reading the times of subset {subset_idx} of BDF "
                f"{bdf_path}: {exc!r}\n === BDF header:\n{bdf_header}"
            ) from exc

    if not midpoints_ns:
        xradio_logger().warning(f"No integrations (subsets) found in BDF {bdf_path}")
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    time_centers = convert_time_asdm_to_unix(np.concatenate(midpoints_ns))
    durations = np.concatenate(durations_ns) / 1e9
    actual_times = convert_time_asdm_to_unix(np.concatenate(actual_times_ns))
    actual_durations = np.concatenate(actual_durations_ns) / 1e9

    return time_centers, durations, actual_times, actual_durations
