import numpy as np
import pyasdm


def ensure_spw_name_conforms(spw_name, spw_id) -> str:
    """
    Create a consistent naming for spectral window name.

    Ensures that the spectral window name follows a consistent format by appending
    the spectral window ID if needed.

    Parameters
    ----------
    spw_name : str or None
        Original spectral window name. If None or empty string, a default name will be generated
    spw_id : int
        ID of the spectral window to append to the name

    Returns
    -------
    str
        The formatted spectral window name in the form "<name>_<id>" or "spw_<id>" if no name provided

    Notes
    -----
    If spw_name is None or empty, returns "spw_<id>"
    If spw_name has content, returns "<spw_name>_<id>"
    """
    if spw_name is None or spw_name == "":
        spw_name = f"spw_{spw_id}"
    else:
        spw_name = f"{spw_name}_{spw_id}"

    return spw_name


def get_spw_name(asdm: pyasdm.ASDM, spw_id: int) -> str | None:
    """Get the name of a spectral window from an ASDM.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object containing the spectral window information
    spw_id : int
        The ID of the spectral window to get the name for

    Returns
    -------
    str or None
        The name of the spectral window if it exists, None otherwise

    Notes
    -----
    Retrieves the name of a spectral window from an ALMA Science Data Model (ASDM)
    by looking up the spectral window ID in the SpectralWindow table. Returns None
    if the spectral window exists but has no name defined.
    """

    spw_tbl = asdm.getSpectralWindow()
    spw_row = spw_tbl.getRowByKey(pyasdm.types.Tag(f"SpectralWindow_{spw_id}"))
    if spw_row.isNameExists():
        name = spw_row.getName()
    else:
        name = None

    return name


def get_spw_frequency_centers(
    asdm: pyasdm.ASDM, spw_id: int, num_chan: int
) -> np.ndarray:
    """
    Get the frequency centers for a given spectral window (spw) from an ASDM dataset.

    This function retrieves the center frequencies for all channels in a spectral window, either by:
    1. Computing them from a start frequency and frequency step size
       (chanFreqStart + i * chanFreqStep, i = 0..numChan-1), when both are present, or
    2. Getting them directly from the channel frequency array (chanFreqArray).

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM dataset object to query
    spw_id : int
        The ID of the spectral window
    num_chan : int
        Expected number of channels in the spectral window. It must match the
        numChan of the spectral window.

    Returns
    -------
    np.ndarray
        Array (float64, Hz) of exactly ``num_chan`` frequency centers, one for each
        channel in the spectral window

    Raises
    ------
    ValueError
        If the spectral window has neither chanFreqStart and chanFreqStep nor
        chanFreqArray
    RuntimeError
        If ``num_chan`` doesn't match the numChan of the spectral window, or the
        number of frequencies in chanFreqArray doesn't match the expected number of
        channels
    """
    spw_tbl = asdm.getSpectralWindow()
    spw_row = spw_tbl.getRowByKey(pyasdm.types.Tag(f"SpectralWindow_{spw_id}"))
    if spw_row.isChanFreqStartExists() and spw_row.isChanFreqStepExists():
        freq_start = spw_row.getChanFreqStart().get()
        freq_step = spw_row.getChanFreqStep().get()
        # Not a float np.arange(start, start + num_chan * step, step): rounding in its
        # stop value can produce num_chan + 1 frequencies.
        frequency_centers = freq_start + freq_step * np.arange(
            num_chan, dtype=np.float64
        )
    elif spw_row.isChanFreqArrayExists():
        frequency_centers = np.array(
            [freq.get() for freq in spw_row.getChanFreqArray()], dtype=np.float64
        )
    else:
        raise ValueError(
            f"Cannot determine the channel frequencies of SpectralWindow_{spw_id}: "
            "it has neither chanFreqStart and chanFreqStep nor chanFreqArray "
            f"(chanFreqStart present: {spw_row.isChanFreqStartExists()}, "
            f"chanFreqStep present: {spw_row.isChanFreqStepExists()})"
        )

    spw_num_chan = spw_row.getNumChan()
    if num_chan != spw_num_chan or len(frequency_centers) != num_chan:
        raise RuntimeError(
            f"Expecting {num_chan} channels for SpectralWindow_{spw_id} but its numChan "
            f"is {spw_num_chan} and {len(frequency_centers)} channel frequencies were "
            "found"
        )

    return frequency_centers


def get_chan_width(asdm: pyasdm.ASDM, spw_id: int) -> float:
    """Get channel width for a given spectral window in an ASDM dataset.

    This function retrieves the channel width either from the single width value
    or from the first element of the width array in the spectral window table.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        ALMA Science Data Model dataset object
    spw_id : int
        Spectral window ID

    Returns
    -------
    float
        Channel width value for the specified spectral window

    Raises
    ------
    ValueError
        If the spectral window has neither chanWidth nor chanWidthArray (both
        optional).

    Notes
    -----
    The function first tries to get a single channel width value. If that doesn't exist,
    it falls back to getting the first value from the channel width array.
    """
    spw_tbl = asdm.getSpectralWindow()
    spw_row = spw_tbl.getRowByKey(pyasdm.types.Tag(f"SpectralWindow_{spw_id}"))
    if spw_row.isChanWidthExists():
        chan_width = spw_row.getChanWidth().get()
    elif spw_row.isChanWidthArrayExists() and len(spw_row.getChanWidthArray()) > 0:
        chan_width = spw_row.getChanWidthArray()[0].get()
    else:
        raise ValueError(
            f"Cannot determine the channel width of SpectralWindow_{spw_id}: it has "
            "neither chanWidth nor chanWidthArray"
        )

    return chan_width


def get_reference_frame(asdm: pyasdm.ASDM, spw_id: int) -> str:
    """Get the reference frame from an ASDM spectral window.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object containing the spectral window data
    spw_id : int
        Spectral window ID number

    Returns
    -------
    str
        Reference frame of the spectral window. Returns 'TOPO' if no reference frame
        is specified in the ASDM data.
    """
    spw_tbl = asdm.getSpectralWindow()
    spw_row = spw_tbl.getRowByKey(pyasdm.types.Tag(f"SpectralWindow_{spw_id}"))
    if spw_row.isMeasFreqRefExists():
        ref_frame = spw_row.getMeasFreqRef().getName()
    else:
        ref_frame = "TOPO"

    return ref_frame
