"""
Creation of the MSv4 pointing_xds sub-dataset from the Pointing table of an ASDM.
"""

import os
import time
import weakref
from dataclasses import dataclass

import numpy as np
import pyasdm
import xarray as xr

from xradio._utils.dict_helpers import (
    make_sky_coord_measure_attrs,
    make_time_measure_attrs,
)
from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)
from xradio.measurement_set._utils._asdm._utils.pointing_direction_rotation import (
    rotate_offset_to_target,
)
from xradio.measurement_set._utils._asdm._utils.time import (
    ASDM_TIME_FORMAT,
    ASDM_TIME_SCALE,
    convert_time_asdm_to_unix,
)

# Bytes read from a Pointing.xml file to find out whether it is only the XML
# header of a binary table (a "<BulkStoreRef" element near the top).
_XML_HEADER_PROBE_BYTES = 65536

_DIRECTION_COLUMNS = ("target", "offset", "pointingDirection")

# Sample times of different rows/antennas that are at most this far apart are
# taken as the same instant on the common time_pointing axis. pyasdm <= 0.0.7
# reads times with float arithmetic (XML timeInterval parse, and the
# correction of its binary fromBin bug), which leaves errors of up to ~1 us
# per row, so the same instant can differ by ~2 us between two antennas. Real
# pointing sampling intervals are milliseconds (ALMA: 48 ms).
_SAME_TIME_TOLERANCE_NS = 10_000

# Minimal Pointing row used to probe how the installed pyasdm parses the XML
# boolean usePolynomials (see _pyasdm_xml_reads_use_polynomials).
_USE_POLYNOMIALS_PROBE_ROW = """<row>
<timeInterval> 5137194870768000000 24240000000 </timeInterval>
<numSample> 1 </numSample> <encoder> 2 1 2 0.0 0.0 </encoder>
<pointingTracking> true </pointingTracking>
<usePolynomials> {value} </usePolynomials>
<timeOrigin> 5137194858648000000 </timeOrigin> <numTerm> 1 </numTerm>
<pointingDirection> 2 1 2 0.0 0.0 </pointingDirection>
<target> 2 1 2 0.0 0.0 </target> <offset> 2 1 2 0.0 0.0 </offset>
<antennaId> Antenna_0 </antennaId> <pointingModelId> 0 </pointingModelId>
</row>"""


class PointingConversionError(ValueError):
    """
    The ASDM Pointing table cannot be converted to a pointing dataset because a
    row is inconsistent: numbers of values that do not match its numSample, or
    an antenna that is not in the Antenna table.
    """


#: Errors that mean that the Pointing table cannot be converted (rather than
#: errors in the arguments or bugs): inconsistent rows, and polynomial
#: expansions, which are not supported (NotImplementedError).
POINTING_CONVERSION_ERRORS = (PointingConversionError, NotImplementedError)


@dataclass
class _CachedPointing:
    """Whole-table pointing conversion cached per ASDM object."""

    #: (number of Pointing rows, number of Antenna rows) when it was built
    key: tuple[int, int]
    #: pointing dataset for all antennas and times, None when there are no samples
    #: or the conversion failed
    xds: xr.Dataset | None
    #: why the table cannot be converted (one of POINTING_CONVERSION_ERRORS,
    #: without traceback), raised again by later calls; None when it was converted
    error: Exception | None = None


# The whole Pointing table is converted only once per ASDM object (open_asdm
# calls create_pointing_xds once per partition). Weak keys: the cached dataset
# goes away together with the ASDM object.
_pointing_cache: "weakref.WeakKeyDictionary[pyasdm.ASDM, _CachedPointing]" = (
    weakref.WeakKeyDictionary()
)


def create_pointing_xds(
    asdm: pyasdm.ASDM,
    time_range: tuple[float, float] | None = None,
    antenna_names: list[str] | None = None,
) -> xr.Dataset | None:
    """
    Build an xarray Dataset with antenna pointing information extracted from
    an ASDM Pointing table.

    How the MSv4 data_vars are derived from the attributes of the ASDM pointing table:

    - MSv4/POINTING_BEAM = rotate(ASDM/target, ASDM/offset) + correction
      where correction = (ASDM/encoder - ASDM/pointingDirection). This is the
      MSv2 DIRECTION written by importasdm with with_pointing_correction=True
      (see :py:class:`~xradio.measurement_set.schema.PointingXds`), and
      rotate(target, offset) applies the offset in the local frame of the
      target (a zero offset gives the target).
    - MSv4/POINTING_DISH_MEASURED = ASDM/encoder
    - MSv4/POINTING_OVER_THE_TOP = ASDM/overTheTop is not produced (optional
      attribute, not present in usual ALMA ASDMs).

    The time_pointing coordinate is the union of the sample times of all the
    antennas (center of every sample, from sampledTimeInterval when present,
    otherwise numSample equal parts of the row timeInterval), in seconds since
    the Unix epoch. Sample times within 10 us of each other are taken as the
    same instant (pyasdm <= 0.0.7 reads times with errors of up to ~1 us).
    Antennas that have no sample at a given time get NaN.

    The conversion of the whole Pointing table is cached per ``asdm`` object,
    so that repeated calls (one per partition) only select from it. When the
    table cannot be converted, the error is logged once and cached as well:
    later calls raise it again without converting the table again.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        ASDM instance from which the Pointing and Antenna tables are read.
    time_range : tuple[float, float] | None
        If given, ``(start, end)`` in seconds since the Unix epoch (UTC, same
        convention as the time coordinate of the correlated datasets). Only
        the samples whose time is within ``[start, end]`` are kept.
    antenna_names : list[str] | None
        If given, only keep these antennas (for example the antennas of a
        partition).

    Returns
    -------
    xr.Dataset | None
        Dataset with the pointing data variables (POINTING_BEAM and
        POINTING_DISH_MEASURED, dims ``(time_pointing, antenna_name,
        local_sky_dir_label)``), or None (and a warning is logged) when the
        Pointing table has no rows or no sample is selected.

    Raises
    ------
    TypeError
        If ``asdm`` is None.
    ValueError
        If ``time_range`` is not a ``(start, end)`` pair with start <= end.
    PointingConversionError
        (a ValueError) If a Pointing row is inconsistent (number of samples,
        unknown antenna).
    NotImplementedError
        If the Pointing table uses polynomial expansions with more than one
        term (``usePolynomials``), which are not supported. See
        ``_read_pointing_row`` for how polynomial rows are detected.

    The last two (``POINTING_CONVERSION_ERRORS``) mean that the Pointing table
    cannot be converted.
    """
    if asdm is None:
        raise TypeError("create_pointing_xds expected a pyasdm.ASDM, got NoneType")

    time_start = time.time()
    full_xds = _get_full_pointing_xds(asdm)
    if full_xds is None:
        xradio_logger().warning(
            "The ASDM Pointing table has no rows (or no samples), pointing_xds "
            "will not be created"
        )
        return None

    xds = full_xds
    if time_range is not None:
        range_start, range_end = _check_time_range(time_range)
        times = xds.coords["time_pointing"].values
        first = np.searchsorted(times, range_start, side="left")
        last = np.searchsorted(times, range_end, side="right")
        xds = xds.isel(time_pointing=slice(first, last))

    if antenna_names is not None:
        wanted = {str(name) for name in antenna_names}
        keep = [str(name) in wanted for name in xds.coords["antenna_name"].values]
        xds = xds.isel(antenna_name=np.flatnonzero(keep))

    # Drop the antennas without any sample in the selection, and the times
    # that are then left without any sample
    valid = ~np.isnan(xds["POINTING_DISH_MEASURED"].values).all(axis=-1)
    valid_times = valid.any(axis=1)
    valid_antennas = valid.any(axis=0)
    if not (valid_times.all() and valid_antennas.all()):
        xds = xds.isel(
            time_pointing=np.flatnonzero(valid_times),
            antenna_name=np.flatnonzero(valid_antennas),
        )
    if xds.sizes["time_pointing"] == 0 or xds.sizes["antenna_name"] == 0:
        xradio_logger().warning(
            "No ASDM Pointing samples in the requested selection "
            f"({time_range=}, {antenna_names=}), pointing_xds will not be created"
        )
        return None

    # Copy, so that changes to the returned dataset cannot alter the cache
    xds = xds.copy(deep=True)

    xradio_logger().debug(
        f"create_pointing_xds() took {time.time() - time_start:0.2f} s"
    )

    return xds


def _check_time_range(time_range: tuple[float, float]) -> tuple[float, float]:
    try:
        range_start, range_end = (float(value) for value in time_range)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"time_range must be a (start, end) pair of unix times, got {time_range!r}"
        ) from exc
    if not range_start <= range_end:
        raise ValueError(
            f"time_range must be a (start, end) pair with start <= end, got {time_range!r}"
        )
    return range_start, range_end


def _get_full_pointing_xds(asdm: pyasdm.ASDM) -> xr.Dataset | None:
    """
    Pointing dataset for the whole Pointing table (all antennas and times),
    converted once per ASDM object and cached. The cache entry is rebuilt if
    the number of rows of the Pointing or Antenna tables changes.

    A conversion that fails with one of POINTING_CONVERSION_ERRORS is logged
    (once) and cached too: later calls raise the same error again, without
    converting the table or logging again.
    """
    pointing_rows = asdm.getPointing().get()
    antenna_df = exp_asdm_table_to_df(asdm, "Antenna", ["antennaId", "name"])
    key = (len(pointing_rows), len(antenna_df))

    try:
        cached = _pointing_cache.get(asdm)
    except TypeError:
        # Object that cannot be weakly referenced: no caching
        cached = None
    if cached is not None and cached.key == key:
        if cached.error is not None:
            raise _detached_copy(cached.error)
        return cached.xds

    time_start = time.time()
    try:
        xds = _build_full_pointing_xds(asdm, pointing_rows, antenna_df)
    except POINTING_CONVERSION_ERRORS as exc:
        xradio_logger().error(
            f"The ASDM Pointing table ({len(pointing_rows)} rows) cannot be "
            f"converted, pointing_xds will not be created. {type(exc).__name__}: "
            f"{exc}"
        )
        # without the traceback, which would keep the conversion data alive
        _cache_pointing(
            asdm, _CachedPointing(key=key, xds=None, error=_detached_copy(exc))
        )
        raise
    _cache_pointing(asdm, _CachedPointing(key=key, xds=xds))
    xradio_logger().info(
        f"Converted the ASDM Pointing table ({len(pointing_rows)} rows) in "
        f"{time.time() - time_start:0.2f} s"
    )

    return xds


def _cache_pointing(asdm: pyasdm.ASDM, entry: _CachedPointing) -> None:
    try:
        _pointing_cache[asdm] = entry
    except TypeError:
        # Object that cannot be weakly referenced: no caching
        pass


def _detached_copy(error: Exception) -> Exception:
    """A new exception of the same type and arguments, without traceback,
    cause or context (the POINTING_CONVERSION_ERRORS have a message only)."""
    return type(error)(*error.args)


@dataclass
class _RowSamples:
    """Samples of one Pointing row, as arrays."""

    antenna_id: int
    #: sample time centers, int64 ASDM nanoseconds
    times: np.ndarray
    #: (num_sample, 2) arrays, radians
    encoder: np.ndarray
    target: np.ndarray
    offset: np.ndarray
    pointing_direction: np.ndarray


def _build_full_pointing_xds(
    asdm: pyasdm.ASDM, pointing_rows: list, antenna_df
) -> xr.Dataset | None:
    if not pointing_rows:
        return None

    fix_frombin_start = _pointing_times_need_frombin_fix(asdm)
    if fix_frombin_start:
        xradio_logger().warning(
            "The installed pyasdm (<= 0.0.7) halves the start times of "
            "ArrayTimeInterval values read from binary tables "
            "(ArrayTimeInterval.fromBin). Correcting the times of the binary "
            "Pointing table, which recovers them to within ~1 us."
        )
    use_polynomials_reliable = _use_polynomials_flag_reliable(asdm)

    row_samples = [
        _read_pointing_row(row, fix_frombin_start, use_polynomials_reliable)
        for row in pointing_rows
    ]
    row_samples = [samples for samples in row_samples if len(samples.times) > 0]
    if not row_samples:
        return None

    antenna_names_by_id = {
        int(antenna_id): str(name)
        for antenna_id, name in zip(
            antenna_df["antennaId"], antenna_df["name"], strict=True
        )
    }
    antenna_ids = sorted({samples.antenna_id for samples in row_samples})
    missing = [ant_id for ant_id in antenna_ids if ant_id not in antenna_names_by_id]
    if missing:
        raise PointingConversionError(
            f"The ASDM Pointing table refers to antennaId(s) {missing} that are not in "
            "the Antenna table"
        )
    antenna_index_by_id = {ant_id: idx for idx, ant_id in enumerate(antenna_ids)}

    sample_times = np.concatenate([samples.times for samples in row_samples])
    sample_antenna = np.concatenate(
        [
            np.full(len(samples.times), antenna_index_by_id[samples.antenna_id])
            for samples in row_samples
        ]
    )

    def concat(name: str) -> np.ndarray:
        return np.concatenate([getattr(samples, name) for samples in row_samples])

    encoder = concat("encoder")
    beam = (
        rotate_offset_to_target(concat("target"), concat("offset"))
        + encoder
        - concat("pointing_direction")
    )

    # Common time axis: union of the sample times of all the antennas, times
    # within _SAME_TIME_TOLERANCE_NS of each other being the same instant.
    # Each antenna's samples are placed at their own times, NaN where an
    # antenna has no sample.
    time_axis_ns, time_index = _merge_sample_times(
        sample_times, _SAME_TIME_TOLERANCE_NS
    )
    num_antenna = len(antenna_ids)
    flat_index = time_index * num_antenna + sample_antenna
    _, first_samples = np.unique(flat_index, return_index=True)
    if len(first_samples) < len(flat_index):
        xradio_logger().warning(
            f"The ASDM Pointing table has {len(flat_index) - len(first_samples)} "
            "samples with the same time (within "
            f"{_SAME_TIME_TOLERANCE_NS / 1000:g} us) as another sample of the "
            "same antenna (overlapping rows). Keeping only the first one."
        )

    shape = (len(time_axis_ns), num_antenna, 2)
    beam_grid = np.full(shape, np.nan, dtype=np.float64)
    encoder_grid = np.full(shape, np.nan, dtype=np.float64)
    grid_index = (time_index[first_samples], sample_antenna[first_samples])
    beam_grid[grid_index] = beam[first_samples]
    encoder_grid[grid_index] = encoder[first_samples]

    time_attrs = make_time_measure_attrs(
        "s", ASDM_TIME_SCALE, time_format=ASDM_TIME_FORMAT
    )
    time_attrs["type"] = "time_pointing"
    coords = {
        # int64 ASDM ns -> unix s (epoch shift in integer ns, K1)
        "time_pointing": (
            "time_pointing",
            convert_time_asdm_to_unix(time_axis_ns.astype(np.int64)),
            time_attrs,
        ),
        "antenna_name": (
            "antenna_name",
            [antenna_names_by_id[ant_id] for ant_id in antenna_ids],
        ),
        "local_sky_dir_label": ("local_sky_dir_label", ["az", "alt"]),
    }

    dims = ("time_pointing", "antenna_name", "local_sky_dir_label")
    direction_attrs = make_sky_coord_measure_attrs("rad", "altaz")
    data_vars = {
        "POINTING_BEAM": (dims, beam_grid, direction_attrs),
        "POINTING_DISH_MEASURED": (dims, encoder_grid, dict(direction_attrs)),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs={"type": "pointing"})


def _merge_sample_times(
    sample_times: np.ndarray, tolerance_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Common time axis of a set of sample times, like ``np.unique(sample_times,
    return_inverse=True)`` but with near-equal times merged.

    The sorted times are split into groups wherever two consecutive times are
    more than ``tolerance_ns`` apart. Each group gives one time of the axis,
    the (rounded) mean of its times, so identical times are kept exactly.

    Parameters
    ----------
    sample_times : np.ndarray
        int64 sample times (ns), any order, at least one.
    tolerance_ns : int
        Largest difference (ns) between consecutive sorted times of a group.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The sorted int64 time axis, and for every sample the index of its
        time in the axis.
    """
    sample_times = np.asarray(sample_times, dtype=np.int64)
    order = np.argsort(sample_times, kind="stable")
    sorted_times = sample_times[order]
    new_group = np.empty(len(sorted_times), dtype=bool)
    new_group[0] = True
    np.greater(np.diff(sorted_times), tolerance_ns, out=new_group[1:])
    group_of_sorted = np.cumsum(new_group) - 1
    group_starts = np.flatnonzero(new_group)
    first_times = sorted_times[group_starts]
    # offsets from the first time of the group, small: the sums cannot overflow
    offsets = sorted_times - first_times[group_of_sorted]
    sums = np.add.reduceat(offsets, group_starts)
    counts = np.diff(np.append(group_starts, len(sorted_times)))
    time_axis = first_times + (2 * sums + counts) // (2 * counts)
    time_index = np.empty_like(group_of_sorted)
    time_index[order] = group_of_sorted
    return time_axis, time_index


def _read_pointing_row(
    row: pyasdm.PointingRow,
    fix_frombin_start: bool,
    use_polynomials_reliable: bool = False,
) -> _RowSamples:
    """
    Read the samples of one Pointing row.

    encoder always has numSample values. target, offset and pointingDirection
    have numTerm values: numSample sampled values, or polynomial coefficients
    when usePolynomials. Polynomial expansions with one term (constant over
    the row) are supported, those with more terms raise NotImplementedError.

    When ``use_polynomials_reliable`` (see _use_polynomials_flag_reliable),
    getUsePolynomials() tells samples and coefficients apart, so that
    coefficients are not taken as samples when numTerm == numSample, and rows
    that are not polynomial but have neither 1 nor numSample values raise
    PointingConversionError. Otherwise (XML Pointing tables read by pyasdm <=
    0.0.7, which parses the flag with bool(text) so "false" reads as True)
    only the number of values is used: a polynomial with exactly numSample
    terms then cannot be detected and is read as samples, and other numbers of
    values raise NotImplementedError (polynomial or inconsistent row).
    Either way the table cannot be converted (POINTING_CONVERSION_ERRORS).
    """
    antenna_id = int(row.getAntennaId().getTagValue())
    num_sample = int(row.getNumSample())
    times = _row_sample_times_ns(row, num_sample, fix_frombin_start)

    encoder = _row_angles(row, "encoder")
    if len(encoder) != num_sample:
        raise PointingConversionError(
            f"Inconsistent ASDM Pointing row (antennaId={antenna_id}, time "
            f"{_describe_row_time(row)}): encoder has {len(encoder)} values but "
            f"numSample={num_sample}"
        )

    use_polynomials = use_polynomials_reliable and bool(row.getUsePolynomials())
    directions = {}
    for name in _DIRECTION_COLUMNS:
        values = _row_angles(row, name)
        if len(values) == 1:
            # constant over the row (single term)
            directions[name] = np.repeat(values, num_sample, axis=0)
        elif len(values) == num_sample and not use_polynomials:
            directions[name] = values
        elif use_polynomials_reliable and not use_polynomials:
            raise PointingConversionError(
                f"Inconsistent ASDM Pointing row (antennaId={antenna_id}, time "
                f"{_describe_row_time(row)}): usePolynomials is false but {name} "
                f"has {len(values)} values for numSample={num_sample}"
            )
        elif use_polynomials:
            raise NotImplementedError(
                f"ASDM Pointing row (antennaId={antenna_id}, time "
                f"{_describe_row_time(row)}) gives {name} as {len(values)} terms "
                f"for numSample={num_sample} (numTerm={row.getNumTerm()}, "
                "usePolynomials=True). Pointing directions given as polynomial "
                "expansions (usePolynomials) with more than one term are not "
                "supported."
            )
        else:
            # The flag cannot be read (see _use_polynomials_flag_reliable): not
            # shown, as it may be misparsed
            raise NotImplementedError(
                f"ASDM Pointing row (antennaId={antenna_id}, time "
                f"{_describe_row_time(row)}) gives {name} as {len(values)} values "
                f"for numSample={num_sample} (numTerm={row.getNumTerm()}). This is "
                "either a polynomial expansion (usePolynomials) with more than one "
                "term, which is not supported, or an inconsistent row (the "
                "usePolynomials flag of this Pointing table cannot be read reliably "
                "with the installed pyasdm)."
            )

    return _RowSamples(
        antenna_id=antenna_id,
        times=times,
        encoder=encoder,
        target=directions["target"],
        offset=directions["offset"],
        pointing_direction=directions["pointingDirection"],
    )


def _row_angles(row: pyasdm.PointingRow, name: str) -> np.ndarray:
    """
    A 2D list of Angle attribute of a Pointing row as a (n, 2) float64 array
    (radians).

    The pyasdm getters return deep copies of the lists of Angle objects, which
    dominate the cost of converting big Pointing tables. The stored lists are
    read directly when available (they are not modified here), with the
    getter as fall back.
    """
    values = getattr(row, f"_{name}", None)
    if values is None:
        values = getattr(row, f"get{name[0].upper()}{name[1:]}")()
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(
        [[angle.get() for angle in pair] for pair in values], dtype=np.float64
    ).reshape(-1, 2)


def _row_sample_times_ns(
    row: pyasdm.PointingRow, num_sample: int, fix_frombin_start: bool
) -> np.ndarray:
    """
    Centers of the samples of a Pointing row, as int64 ASDM nanoseconds.

    From sampledTimeInterval (one interval per sample) when present, otherwise
    the row timeInterval is split in numSample equal parts. Integer arithmetic
    is used throughout (ASDM times ~5e18 ns are beyond float64 precision).
    """
    if row.isSampledTimeIntervalExists():
        sampled = getattr(row, "_sampledTimeInterval", None)
        if sampled is None:
            sampled = row.getSampledTimeInterval()
        if len(sampled) != num_sample:
            raise PointingConversionError(
                f"Inconsistent ASDM Pointing row (antennaId="
                f"{row.getAntennaId().getTagValue()}, time {_describe_row_time(row)}): "
                f"sampledTimeInterval has {len(sampled)} values but "
                f"numSample={num_sample}"
            )
        centers = []
        for interval in sampled:
            start, duration = _interval_start_duration(interval, fix_frombin_start)
            centers.append(start + duration // 2)
        return np.array(centers, dtype=np.int64)

    start, duration = _interval_start_duration(row.getTimeInterval(), fix_frombin_start)
    if num_sample <= 0:
        return np.empty(0, dtype=np.int64)
    sample_idx = np.arange(num_sample, dtype=np.int64)
    # start + (i + 1/2) * duration / num_sample
    return start + ((2 * sample_idx + 1) * duration) // (2 * num_sample)


def _interval_start_duration(
    interval: pyasdm.types.ArrayTimeInterval, fix_frombin_start: bool
) -> tuple[int, int]:
    start = int(interval.getStart().get())
    duration = int(interval.getDuration().get())
    if fix_frombin_start:
        start = _undo_frombin_start_halving(start, duration)
    return start, duration


def _describe_row_time(row: pyasdm.PointingRow) -> str:
    interval = row.getTimeInterval()
    return f"start={interval.getStart().get()} ns, duration={interval.getDuration().get()} ns"


def _undo_frombin_start_halving(start_ns: int, duration_ns: int) -> int:
    """
    Recover the start time of an interval read by a pyasdm version whose
    ArrayTimeInterval.fromBin returns start = int((midpoint - duration) / 2)
    instead of midpoint - duration / 2. The (float) halving loses up to
    ~0.5 microsecond of the midpoint, which depends on each row's midpoint
    and duration (see _SAME_TIME_TOLERANCE_NS).
    """
    midpoint = 2 * start_ns + duration_ns
    return midpoint - duration_ns // 2


class _TwoLongsReader:
    """Minimal stand-in for a pyasdm EndianInput, gives two long values."""

    def __init__(self, first: int, second: int):
        self._values = [first, second]

    def readLong(self) -> int:
        return self._values.pop(0)


def _pyasdm_frombin_halves_start() -> bool:
    """
    Probe the installed pyasdm: True if ArrayTimeInterval.fromBin converts the
    (midpoint, duration) read from binary tables into start = int((midpoint -
    duration) / 2) (bug of pyasdm <= 0.0.7) instead of midpoint - duration / 2.
    False when fromBin is correct, or when it reads (start, duration) because
    ArrayTimeInterval.readStartTimeDurationInBin() is set.
    """
    midpoint = 5_137_194_870_768_000_000  # 2021-09-01, ASDM ns
    duration = 24_240_000_000
    try:
        interval = pyasdm.types.ArrayTimeInterval.fromBin(
            _TwoLongsReader(midpoint, duration)
        )
        start = int(interval.getStart().get())
    except Exception as exc:  # pragma: no cover - defensive, depends on pyasdm
        xradio_logger().debug(
            f"Could not probe pyasdm ArrayTimeInterval.fromBin: {exc}"
        )
        return False
    return start == int((midpoint - duration) / 2)


def _pointing_table_read_from_binary(asdm: pyasdm.ASDM) -> bool:
    """
    Whether pyasdm loads the Pointing table of this ASDM from its binary file
    (Pointing.bin), following the same rules as PointingTable.setFromFile: a
    Pointing.xml with a "<BulkStoreRef" element (header of a binary table), or
    no Pointing.xml but a Pointing.bin. False for ASDMs not read from a
    directory (tables filled in memory).
    """
    directory = asdm.getDirectory() if hasattr(asdm, "getDirectory") else None
    if not isinstance(directory, str) or not directory:
        return False
    xml_path = os.path.join(directory, "Pointing.xml")
    if os.path.exists(xml_path):
        with open(xml_path, "rb") as xml_file:
            header = xml_file.read(_XML_HEADER_PROBE_BYTES)
        return b"<BulkStoreRef" in header
    return os.path.exists(os.path.join(directory, "Pointing.bin"))


def _pointing_times_need_frombin_fix(asdm: pyasdm.ASDM) -> bool:
    """True if the Pointing times of this ASDM are affected by the pyasdm
    fromBin bug (see _pyasdm_frombin_halves_start) and must be corrected."""
    return _pyasdm_frombin_halves_start() and _pointing_table_read_from_binary(asdm)


def _pyasdm_xml_reads_use_polynomials() -> bool:
    """
    Probe the installed pyasdm: True if PointingRow.setFromXML reads the
    boolean usePolynomials correctly ("false" -> False, "true" -> True).
    pyasdm <= 0.0.7 parses it with bool(text), so that "false" reads as True.
    """
    try:
        values = []
        for text in ("false", "true"):
            row = pyasdm.PointingRow(pyasdm.ASDM().getPointing())
            row.setFromXML(_USE_POLYNOMIALS_PROBE_ROW.format(value=text))
            values.append(row.getUsePolynomials())
    except Exception as exc:  # pragma: no cover - defensive, depends on pyasdm
        xradio_logger().debug(f"Could not probe pyasdm PointingRow.setFromXML: {exc}")
        return False
    return not values[0] and bool(values[1])


def _use_polynomials_flag_reliable(asdm: pyasdm.ASDM) -> bool:
    """
    Whether getUsePolynomials() of the Pointing rows of this ASDM gives the
    value stored in the ASDM: always for a binary Pointing table (read with
    readBool, as usual for ALMA), and for other tables (XML, or filled in
    memory) only when the installed pyasdm parses XML booleans correctly.
    """
    return _pointing_table_read_from_binary(asdm) or _pyasdm_xml_reads_use_polynomials()
