"""
Calculations with the shapes of the auto/cross data and flags binary components of
BDFs, and with the positions (row lengths and offsets) of the data of one SPW in
the 1D data trees of those binary components.

Layout of the binary components of one subset (BDF specification, axes listed from
the outermost to the innermost, optional axes in brackets)::

    crossData: [TIM] BAL BAB SPW [BIN] [APC] SPP POL (re, im)
    autoData:  [TIM] ANT BAB SPW [BIN] SPP POL
    flags:     [TIM] BAL ANT BAB [SPW] POL (a block of BAL rows followed by a block
               of ANT rows)

The "rows" of crossData are the baselines and the rows of autoData the antennas.
Every row holds the data of all the SPWs of the BDF, one after the other, so the
data of one SPW is found at a fixed offset of every row. A full-polarization
autoData channel (sdPolProducts XX XY YY) is stored as the 4 floats
XX, Re(XY), Im(XY), YY.
"""

import numbers
from dataclasses import dataclass

from xradio._utils.logging import xradio_logger

#: Value of the atmospheric phase correction loaded when crossData has more than
#: one (CASA importasdm default)
APC_LOADED = "AP_UNCORRECTED"

#: Order of the axes of the BDF binary components, from outermost to innermost
_BDF_AXES_ORDER = (
    "TIM",
    "BAL",
    "ANT",
    "BAB",
    "SPW",
    "SIB",
    "SUB",
    "BIN",
    "APC",
    "SPP",
    "POL",
    "STO",
    "HOL",
)

#: Axes of the data binary components that the loaders can handle
_SUPPORTED_DATA_AXES = {
    "crossData": ("TIM", "BAL", "BAB", "SPW", "BIN", "APC", "SPP", "POL"),
    "autoData": ("TIM", "ANT", "BAB", "SPW", "BIN", "SPP", "POL"),
}


def num_auto_values_per_channel(num_sd_pols: int) -> int:
    """
    Number of float values per channel in autoData.

    Parameters
    ----------
    num_sd_pols : int
        Number of sdPolProducts of the SPW.

    Returns
    -------
    int
        4 for full-polarization autoData (XX, Re(XY), Im(XY), YY), otherwise the
        number of sdPolProducts.
    """
    return 4 if num_sd_pols == 3 else num_sd_pols


def times_per_subset(bdf_descr: dict) -> int:
    """
    Number of integrations (TIM samples) in every subset of a BDF.

    Parameters
    ----------
    bdf_descr : dict
        BDF description (from the BDF header).

    Returns
    -------
    int
        numTime for packed BDFs (dimensionality 0, all integrations in one subset),
        1 otherwise (one integration per subset).
    """
    if bdf_descr["dimensionality"] == 0:
        num_time = int(bdf_descr["num_time"])
        if num_time < 1:
            raise ValueError(
                f"Packed BDF (dimensionality 0) with invalid numTime: {num_time}"
            )
        return num_time

    return 1


def _correlation_mode_name(bdf_descr: dict) -> str:
    return str(bdf_descr["correlation_mode"])


@dataclass(frozen=True)
class BDFSpwLayout:
    """
    Position of the data of one SPW in the crossData and autoData binary components
    of the subsets of one BDF.

    All lengths and offsets count elements (values) of the binary components. In
    crossData the real and imaginary parts are separate elements.

    Attributes
    ----------
    times_per_subset : int
        Integrations (TIM samples) per subset.
    num_cross_baselines : int
        Number of rows of crossData (0 for AUTO_ONLY data).
    num_antennas : int
        Number of rows of autoData.
    num_channels : int
        Number of channels of the SPW.
    num_polarizations : int
        Length of the output polarization axis (crossPolProducts, or sdPolProducts
        for AUTO_ONLY data).
    num_cross_pols : int
        Number of crossPolProducts of the SPW.
    num_sd_pols : int
        Number of sdPolProducts of the SPW.
    cross_row_len : int
        Number of crossData values per baseline (all SPWs, bins and APCs).
    cross_spw_offset : int
        Offset of the selected SPW (and APC) within a crossData row.
    auto_row_len : int
        Number of autoData values per antenna (all SPWs and bins).
    auto_spw_offset : int
        Offset of the SPW within an autoData row.
    scale_factor : float
        Scale factor of the SPW. Integer crossData are divided by it.
    apc_index : int
        Index of the APC loaded (in the BDF APC list).
    real_cross_data_allowed : bool
        Whether crossData may hold real values (one value per polarization product
        instead of (re, im)), as accepted for data not from the CORRELATOR.
    """

    times_per_subset: int
    num_cross_baselines: int
    num_antennas: int
    num_channels: int
    num_polarizations: int
    num_cross_pols: int
    num_sd_pols: int
    cross_row_len: int
    cross_spw_offset: int
    auto_row_len: int
    auto_spw_offset: int
    scale_factor: float
    apc_index: int
    real_cross_data_allowed: bool = False

    @property
    def num_baselines(self) -> int:
        """Length of the output baseline axis (cross baselines followed by autos)."""
        return self.num_cross_baselines + self.num_antennas

    @property
    def num_auto_values(self) -> int:
        """Number of autoData float values per channel."""
        return num_auto_values_per_channel(self.num_sd_pols)

    @property
    def cross_size(self) -> int:
        """Expected size of the crossData binary component of one subset."""
        return self.times_per_subset * self.num_cross_baselines * self.cross_row_len

    @property
    def auto_size(self) -> int:
        """Expected size of the autoData binary component of one subset."""
        return self.times_per_subset * self.num_antennas * self.auto_row_len


def select_apc_index(apc_list: list | None, *, warn: bool = True) -> int:
    """
    Index (in the BDF APC list) of the atmospheric phase correction to load.

    When the crossData have several atmospheric phase corrections, AP_UNCORRECTED
    (``APC_LOADED``) is loaded, as CASA importasdm does by default.

    Parameters
    ----------
    apc_list : list | None
        APC list from the BDF header (pyasdm AtmPhaseCorrection values or their
        names).
    warn : bool, default True
        Log a warning when the only APC is not AP_UNCORRECTED (that APC is then
        loaded). Meant for a report once per partition (when it is opened): the
        data loaders, which run for every BDF and chunk, pass False.

    Returns
    -------
    int
        Index of AP_UNCORRECTED, or 0 when there is only one APC (or none).

    Raises
    ------
    NotImplementedError
        If there are several APCs and none is AP_UNCORRECTED.
    """
    apc_names = [str(apc) for apc in apc_list or []]
    if APC_LOADED in apc_names:
        return apc_names.index(APC_LOADED)
    if len(apc_names) <= 1:
        if apc_names and warn:
            xradio_logger().warning(
                "The crossData have only the atmospheric phase correction "
                f"{apc_names[0]}, not {APC_LOADED}. Loading {apc_names[0]} data."
            )
        return 0

    raise NotImplementedError(
        f"Cannot select the atmospheric phase correction to load. The crossData have "
        f"{apc_names=}, which does not include {APC_LOADED} (only {APC_LOADED} is "
        "supported when there are several)."
    )


def check_data_axes(axes: dict[str, list[str]] | None):
    """
    Checks that the axes of the crossData and autoData binary components (as given
    in the BDF header) can be handled by the loaders.

    Parameters
    ----------
    axes : dict[str, list[str]] | None
        Axes names of the binary components, by component name. None to skip the
        checks.

    Raises
    ------
    NotImplementedError
        If an axis is not supported or the axes are not in the BDF order.
    """
    if not axes:
        return

    for component, supported in _SUPPORTED_DATA_AXES.items():
        component_axes = [str(axis) for axis in axes.get(component) or []]
        unsupported = [axis for axis in component_axes if axis not in supported]
        if unsupported:
            raise NotImplementedError(
                f"Loading {component} with axes {unsupported} is not supported "
                f"({component} axes: {component_axes})."
            )
        positions = [_BDF_AXES_ORDER.index(axis) for axis in component_axes]
        if positions != sorted(positions):
            raise NotImplementedError(
                f"Unexpected order of the {component} axes: {component_axes}."
            )


def calc_bdf_spw_layout(
    bdf_descr: dict, baseband_spw_idxs: tuple[int, int]
) -> BDFSpwLayout:
    """
    Calculates where the data of one SPW are in the crossData and autoData binary
    components of the subsets of a BDF.

    Every SPW of the BDF takes ``numBin * numAPC * numSpectralPoint *
    len(crossPolProducts) * 2`` values per baseline in crossData and ``numBin *
    numSpectralPoint * num_auto_values`` per antenna in autoData. Of the
    atmospheric phase corrections (APC) present in crossData, AP_UNCORRECTED is
    selected (or the only APC, whatever it is, without warning: see
    select_apc_index).

    The loaders calculate the layout once per BDF load and pass it to the
    per-subset functions.

    Parameters
    ----------
    bdf_descr : dict
        BDF description (from the BDF header). An optional "axes" entry (dict of
        axes names by binary component) enables additional checks.
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband and index of the SPW within the baseband.

    Returns
    -------
    BDFSpwLayout
        Layout of the SPW data.

    Raises
    ------
    NotImplementedError
        If the SPW has numBin > 1, the data axes are not supported, or crossData
        have several APCs but not AP_UNCORRECTED.
    ValueError
        If the BDF description is inconsistent.
    """
    check_data_axes(bdf_descr.get("axes"))

    correlation_mode = _correlation_mode_name(bdf_descr)
    if correlation_mode == "CROSS_AND_AUTO":
        cross_present = True
    elif correlation_mode == "AUTO_ONLY":
        cross_present = False
    else:
        raise ValueError(f"Unsupported correlation mode: {correlation_mode}")

    baseband_idx, spw_idx = baseband_spw_idxs
    basebands = bdf_descr["basebands"]
    try:
        spw_descr = basebands[baseband_idx]["spectralWindows"][spw_idx]
    except IndexError as exc:
        raise ValueError(
            f"SPW index {spw_idx} of baseband index {baseband_idx} not found in the "
            "BDF basebands."
        ) from exc

    num_bin = spw_descr.get("numBin") or 1
    if num_bin > 1:
        raise NotImplementedError(
            f"Loading data with numBin > 1 (numBin={num_bin} in baseband "
            f"{basebands[baseband_idx].get('name')}, SPW index {spw_idx}) is not "
            "supported."
        )

    num_antennas = int(bdf_descr["num_antenna"])
    if cross_present:
        num_cross_baselines = num_antennas * (num_antennas - 1) // 2
        apc_list = bdf_descr.get("apc") or []
        if not isinstance(apc_list, list | tuple):
            apc_list = [apc_list]
        num_apc = len(apc_list) or 1
        # No warning here: the layout is calculated for every BDF loaded (for every
        # chunk). The APC loaded is recorded (and can be reported) once per
        # partition, when it is opened.
        apc_index = select_apc_index(apc_list, warn=False)
    else:
        num_cross_baselines = 0
        num_apc = 1
        apc_index = 0

    num_channels = int(spw_descr["numSpectralPoint"])
    num_cross_pols = len(spw_descr["crossPolProducts"]) if cross_present else 0
    num_sd_pols = len(spw_descr["sdPolProducts"])
    if cross_present:
        num_polarizations = num_cross_pols
        if num_sd_pols != num_cross_pols and not (
            num_sd_pols == 3 and num_cross_pols == 4
        ):
            raise ValueError(
                f"Unsupported combination of {num_cross_pols} crossPolProducts and "
                f"{num_sd_pols} sdPolProducts."
            )
    else:
        num_polarizations = num_sd_pols
    if num_polarizations < 1:
        raise ValueError(f"SPW without polarization products: {spw_descr=}")

    cross_row_len = cross_spw_offset = auto_row_len = auto_spw_offset = 0
    for bb_idx, baseband in enumerate(basebands):
        for idx, spw in enumerate(baseband["spectralWindows"]):
            spw_nbin = spw.get("numBin") or 1
            spw_nchan = int(spw["numSpectralPoint"])
            if (bb_idx, idx) == (baseband_idx, spw_idx):
                cross_spw_offset = cross_row_len + apc_index * (
                    num_channels * num_cross_pols * 2
                )
                auto_spw_offset = auto_row_len
            if cross_present:
                cross_row_len += (
                    spw_nbin * num_apc * spw_nchan * len(spw["crossPolProducts"]) * 2
                )
            auto_row_len += (
                spw_nbin
                * spw_nchan
                * num_auto_values_per_channel(len(spw["sdPolProducts"]))
            )

    scale_factor = spw_descr.get("scaleFactor") or 1

    return BDFSpwLayout(
        times_per_subset=times_per_subset(bdf_descr),
        num_cross_baselines=num_cross_baselines,
        num_antennas=num_antennas,
        num_channels=num_channels,
        num_polarizations=num_polarizations,
        num_cross_pols=num_cross_pols,
        num_sd_pols=num_sd_pols,
        cross_row_len=cross_row_len,
        cross_spw_offset=cross_spw_offset,
        auto_row_len=auto_row_len,
        auto_spw_offset=auto_spw_offset,
        scale_factor=float(scale_factor),
        apc_index=apc_index,
        real_cross_data_allowed=str(bdf_descr.get("processor_type")) != "CORRELATOR",
    )


@dataclass(frozen=True)
class BDFSelection:
    """
    Index ranges [start, stop) selected along the 4 dimensions (time, baseline,
    frequency, polarization) of the data of one SPW of a BDF.

    Attributes
    ----------
    time : tuple[int, int | None]
        Time (integration) range. stop is None for "until the last integration".
    baseline : tuple[int, int]
        Baseline range (cross baselines followed by autos, or antennas).
    frequency : tuple[int, int]
        Channel range.
    polarization : tuple[int, int]
        Polarization range.
    """

    time: tuple[int, int | None]
    baseline: tuple[int, int]
    frequency: tuple[int, int]
    polarization: tuple[int, int]

    def cross_range(self, num_cross_baselines: int) -> tuple[int, int] | None:
        """Range of crossData rows (baselines) selected, None if none."""
        start, stop = self.baseline
        if start >= num_cross_baselines:
            return None
        return start, min(stop, num_cross_baselines)

    def auto_range(self, num_cross_baselines: int) -> tuple[int, int] | None:
        """Range of autoData rows (antennas) selected, None if none."""
        start, stop = self.baseline
        if stop <= num_cross_baselines:
            return None
        return max(start - num_cross_baselines, 0), stop - num_cross_baselines


def _dim_range(
    key: slice | int | None, dim_len: int | None, dim_name: str
) -> tuple[int, int | None]:
    """
    Turns one element of an array_slice (slice, int or None) into a [start, stop)
    range. Int keys give a range of length 1 (the dimension is kept). dim_len None
    means unknown length (stop can then be None: until the end).
    """
    if key is None:
        return 0, dim_len

    if isinstance(key, numbers.Integral):
        index = int(key)
        if index < 0:
            if dim_len is None:
                raise ValueError(f"Negative {dim_name} index {index} not supported")
            index += dim_len
        if index < 0 or (dim_len is not None and index >= dim_len):
            raise IndexError(
                f"{dim_name} index {key} out of range (dimension length {dim_len})"
            )
        return index, index + 1

    if isinstance(key, slice):
        if key.step not in (None, 1):
            raise ValueError(
                f"Only step 1 is supported in {dim_name} slices, got {key=}"
            )
        if dim_len is None:
            start = key.start or 0
            stop = key.stop
            if start < 0 or (stop is not None and stop < 0):
                raise ValueError(
                    f"Negative {dim_name} slice bounds not supported, got {key=}"
                )
        else:
            start, stop, _ = key.indices(dim_len)
        if stop is not None and stop <= start:
            raise ValueError(f"Empty {dim_name} selection: {key=}")
        return start, stop

    raise TypeError(f"Unexpected type of {dim_name} key: {type(key)} ({key=})")


def normalize_bdf_selection(
    array_slice: tuple[slice | int | None, ...] | None,
    layout: BDFSpwLayout,
    time_len: int | None = None,
) -> BDFSelection:
    """
    Turns an array_slice (time, baseline, frequency, polarization) of slices, ints
    or None into explicit index ranges.

    Parameters
    ----------
    array_slice : tuple[slice | int | None, ...] | None
        Selection along (time, baseline, frequency, polarization). Slices must have
        step 1. Int keys select one index but keep the dimension. None or a missing
        element selects the whole dimension.
    layout : BDFSpwLayout
        Layout of the SPW data (gives the length of the dimensions).
    time_len : int | None
        Length of the time dimension, None if unknown.

    Returns
    -------
    BDFSelection
        Selected ranges.
    """
    keys = tuple(array_slice) if array_slice else ()
    keys = keys + (None,) * (4 - len(keys))
    if len(keys) != 4:
        raise ValueError(f"Expected a selection of 4 dimensions, got {array_slice=}")

    return BDFSelection(
        time=_dim_range(keys[0], time_len, "time"),
        baseline=_dim_range(keys[1], layout.num_baselines, "baseline"),
        frequency=_dim_range(keys[2], layout.num_channels, "frequency"),
        polarization=_dim_range(keys[3], layout.num_polarizations, "polarization"),
    )
