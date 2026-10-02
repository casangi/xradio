"""
Opening of BDFs (with errors that name the BDF) and sanity checks of various fields
of the BDF description dict (BDF header metadata)
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pyasdm

from xradio.measurement_set._utils._asdm._utils._bdf.shapes import (
    num_auto_values_per_channel,
    select_apc_index,
)

#: Axes of the crossData/autoData binary components that the loaders do not handle
UNSUPPORTED_DATA_AXES = ("SIB", "SUB", "STO", "HOL")


class BDFOpenError(RuntimeError, pyasdm.exceptions.BDFReaderException):
    """
    A BDF cannot be opened or its global header cannot be parsed (missing,
    unreadable, empty, truncated or corrupt file). The message names the BDF.

    It is a RuntimeError, and also a pyasdm BDFReaderException for callers that
    catch the errors of the pyasdm BDF reader.
    """


@contextmanager
def open_bdf(bdf_path: str) -> Iterator[pyasdm.bdf.BDFReader]:
    """
    Open a BDF with a pyasdm BDFReader (which reads and parses the global header),
    closing it on exit.

    Parameters
    ----------
    bdf_path : str
        Path of the BDF.

    Yields
    ------
    pyasdm.bdf.BDFReader
        The opened reader, positioned before the first subset.

    Raises
    ------
    BDFOpenError
        If the BDF cannot be opened or its header cannot be parsed. The message
        names the BDF (pyasdm's own messages do not when the header is corrupt).
    """
    bdf_path = str(bdf_path)
    bdf_reader = pyasdm.bdf.BDFReader()
    try:
        bdf_reader.open(bdf_path)
    except Exception as exc:
        # the reader may have opened the file before failing to parse the header
        bdf_reader.close()
        raise BDFOpenError(
            f"Cannot open the BDF {bdf_path} or parse its header. Details: {exc!r}"
        ) from exc
    try:
        yield bdf_reader
    finally:
        bdf_reader.close()


def _in_bdf(bdf_path: str | None) -> str:
    """Suffix naming the BDF in error messages (empty if not known)."""
    return "" if bdf_path is None else f" in BDF {bdf_path}"


def check_basebands(basebands: list[dict], bdf_path: str | None = None):
    """
    Check the number of basebands of a BDF.

    Parameters
    ----------
    basebands : list[dict]
        Basebands list from the BDF header.
    bdf_path : str | None
        Path of the BDF (for error messages).

    Raises
    ------
    RuntimeError
        If the BDF does not have 1 to 4 basebands.
    """
    # An example of 2 basebands: uid___A002_X9bb85e_Xcb (I think they are rare)
    if len(basebands) not in [1, 2, 3, 4]:
        raise RuntimeError(
            f"Unexpected number of basebands{_in_bdf(bdf_path)}: "
            f"{len(basebands)=}, {basebands=}"
        )


def check_correlation_mode(
    correlation_mode: pyasdm.enumerations.CorrelationMode, bdf_path: str | None = None
):
    """
    Check that the correlation mode is supported (CROSS_AND_AUTO or AUTO_ONLY).

    Parameters
    ----------
    correlation_mode : pyasdm.enumerations.CorrelationMode
        Correlation mode from the BDF header.
    bdf_path : str | None
        Path of the BDF (for error messages).

    Raises
    ------
    RuntimeError
        For CROSS_ONLY data.
    """
    if correlation_mode == pyasdm.enumerations.CorrelationMode.CROSS_ONLY:
        raise RuntimeError(
            f"Unsupported correlation mode {correlation_mode}{_in_bdf(bdf_path)} "
            "(only CROSS_AND_AUTO and AUTO_ONLY are supported)"
        )


def required_binary_components(
    correlation_mode: pyasdm.enumerations.CorrelationMode,
) -> list[str]:
    """
    Data binary components that a BDF must have, given its correlation mode.

    Flags are not required: when a BDF has no flags binary component nothing is
    flagged.

    Parameters
    ----------
    correlation_mode : pyasdm.enumerations.CorrelationMode
        Correlation mode from the BDF header.

    Returns
    -------
    list[str]
        ["crossData", "autoData"] for CROSS_AND_AUTO, ["autoData"] for AUTO_ONLY,
        ["crossData"] for CROSS_ONLY.
    """
    if correlation_mode == pyasdm.enumerations.CorrelationMode.AUTO_ONLY:
        return ["autoData"]
    if correlation_mode == pyasdm.enumerations.CorrelationMode.CROSS_ONLY:
        return ["crossData"]
    return ["crossData", "autoData"]


def ensure_presence_binary_components(
    data_array_names: list[str], binary_types: list[str], bdf_path: str
):
    """
    Check that a BDF has all the binary components required.

    Parameters
    ----------
    data_array_names : list[str]
        Names of the binary components required.
    binary_types : list[str]
        Names of the binary components present in the BDF (with a non-zero size in
        the BDF header, see make_bdf_description).
    bdf_path : str
        Path of the BDF (for error messages).

    Raises
    ------
    RuntimeError
        If a required binary component is not present.
    """
    for array_name in data_array_names:
        if array_name not in binary_types:
            raise RuntimeError(
                f"When trying to load data from BDF: {bdf_path}, it does not "
                f"have the binary component {array_name} (it has: {binary_types})"
            )


def exclude_unsupported_axis_names(
    dims: list[str], exclude_also_for_flags: bool = False, bdf_path: str | None = None
):
    """
    Check that the axes of a binary component do not include unsupported axes.

    Parameters
    ----------
    dims : list[str]
        Axes names.
    exclude_also_for_flags : bool
        Also reject the axes that are not supported in the flags (APC, SPP).
    bdf_path : str | None
        Path of the BDF (for error messages).

    Raises
    ------
    RuntimeError
        If unsupported axes are found.
    """
    # This effectively assumes we'll always get "POL" from the last 3 possible axes,
    # from BDF doc: "The final three axes, STO, POL and HOL, also appear at the same
    # level in the axis hierarchy; however, only one of these axes will normally
    # appear for a given binary component type.
    unsupported = ["STO", "HOL"]

    if exclude_also_for_flags:
        unsupported.extend(["APC", "SPP"])

    bad_found = []
    for bad_dim in unsupported:
        if bad_dim in dims:
            bad_found.append(bad_dim)

    if bad_found:
        raise RuntimeError(
            f"Unsupported dimension(s) {bad_found=} in {dims=}{_in_bdf(bdf_path)}"
        )


def check_num_bin(
    basebands: list[dict], baseband_spw_idxs: tuple[int, int], bdf_path: str
):
    """
    Check that the SPW to load does not have several bins (numBin > 1, as in
    switching modes), which is not supported.

    Parameters
    ----------
    basebands : list[dict]
        Basebands list from the BDF header.
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband and index of the SPW within the baseband.
    bdf_path : str
        Path of the BDF (for error messages).

    Raises
    ------
    NotImplementedError
        If the SPW has numBin > 1.
    """
    baseband_idx, spw_idx = baseband_spw_idxs
    spw_descr = basebands[baseband_idx]["spectralWindows"][spw_idx]
    num_bin = spw_descr.get("numBin") or 1
    if num_bin > 1:
        raise NotImplementedError(
            f"Loading data with numBin > 1 is not supported (numBin={num_bin} in "
            f"SPW {spw_idx} of baseband {basebands[baseband_idx].get('name')}, BDF "
            f"{bdf_path})."
        )


def expected_data_sizes(bdf_descr: dict) -> dict[str, list[int]]:
    """
    Sizes (number of values) of the crossData and autoData binary components of
    one subset, as expected from the basebands/SPWs description of a BDF.

    The model is (BDF axes TIM BAL BAB SPW BIN APC SPP POL for crossData, and TIM
    ANT BAB SPW BIN SPP POL for autoData)::

        crossData: ntim * nbaselines * sum_spw(numBin * numAPC * nchan * ncross * 2)
        autoData:  ntim * nantennas * sum_spw(numBin * nchan * nauto_values)

    where ntim is numTime for packed BDFs (1 otherwise), numAPC the length of the
    APC list (1 if empty) and nauto_values 4 for 3 sdPolProducts (XX, Re(XY),
    Im(XY), YY). Data that are not from the CORRELATOR may hold real crossData
    values (no factor 2).

    Parameters
    ----------
    bdf_descr : dict
        BDF description (from make_bdf_description).

    Returns
    -------
    dict[str, list[int]]
        Acceptable sizes, by binary component name.
    """
    num_tim = bdf_descr["num_time"] if bdf_descr["dimensionality"] == 0 else 1
    num_antenna = bdf_descr["num_antenna"]
    num_baselines = num_antenna * (num_antenna - 1) // 2
    num_apc = len(bdf_descr.get("apc") or []) or 1

    cross_values = 0
    auto_values = 0
    for baseband in bdf_descr["basebands"]:
        for spw in baseband["spectralWindows"]:
            num_bin = spw.get("numBin") or 1
            num_chan = spw["numSpectralPoint"]
            cross_values += num_bin * num_apc * num_chan * len(spw["crossPolProducts"])
            auto_values += (
                num_bin
                * num_chan
                * num_auto_values_per_channel(len(spw["sdPolProducts"]))
            )

    cross_sizes = [num_tim * num_baselines * cross_values * 2]
    if bdf_descr["processor_type"] != pyasdm.enumerations.ProcessorType.CORRELATOR:
        cross_sizes.append(num_tim * num_baselines * cross_values)

    return {
        "crossData": cross_sizes,
        "autoData": [num_tim * num_antenna * auto_values],
    }


def check_data_components(
    bdf_descr: dict, baseband_spw_idxs: tuple[int, int], bdf_path: str
) -> int:
    """
    Check that the layout of the crossData/autoData binary components of a BDF can
    be loaded: numBin of the SPW, supported axes, APC selection, and the sizes
    announced in the BDF header consistent with the basebands/SPWs description
    (otherwise the data positions computed by the loaders would be wrong).

    Parameters
    ----------
    bdf_descr : dict
        BDF description (from make_bdf_description, with "axes" and "sizes").
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband and index of the SPW within the baseband.
    bdf_path : str
        Path of the BDF (for error messages).

    Returns
    -------
    int
        Index (in the BDF APC list) of the atmospheric phase correction to load.

    Raises
    ------
    NotImplementedError
        For numBin > 1, unsupported axes, or several APCs without AP_UNCORRECTED.
    RuntimeError
        If the sizes of the data binary components do not match the description.
    """
    check_num_bin(bdf_descr["basebands"], baseband_spw_idxs, bdf_path)

    axes = bdf_descr.get("axes") or {}
    for component in ("crossData", "autoData"):
        component_axes = [str(axis) for axis in axes.get(component) or []]
        unsupported = [axis for axis in component_axes if axis in UNSUPPORTED_DATA_AXES]
        if unsupported:
            raise NotImplementedError(
                f"Loading {component} with axes {unsupported} is not supported "
                f"({component} axes: {component_axes}, BDF {bdf_path})."
            )

    apc_index = select_apc_index(bdf_descr.get("apc") or [], warn=False)

    sizes = bdf_descr.get("sizes") or {}
    expected_sizes = expected_data_sizes(bdf_descr)
    for component in bdf_descr.get("binary_types", []):
        if component not in expected_sizes or component not in sizes:
            continue
        if sizes[component] not in expected_sizes[component]:
            raise RuntimeError(
                f"Unsupported or inconsistent data layout in BDF {bdf_path}: the "
                f"header announces {component} size={sizes[component]} but the "
                f"basebands/SPWs description gives {expected_sizes[component]} "
                f"({component} axes: {axes.get(component)}, "
                f"APC: {[str(apc) for apc in bdf_descr.get('apc') or []]}, "
                f"numTime: {bdf_descr['num_time']}, "
                f"dimensionality: {bdf_descr['dimensionality']})."
            )

    return apc_index
