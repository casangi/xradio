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

    spw_name


def get_spw_frequency_centers(
    asdm: pyasdm.ASDM, spw_id: int, num_chan: int
) -> np.ndarray:
    """
    Get the frequency centers for a given spectral window (spw) from an ASDM dataset.

    This function retrieves the center frequencies for all channels in a spectral window, either by:
    1. Computing them from a start frequency and frequency step size, or
    2. Getting them directly from the channel frequency array.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM dataset object to query
    spw_id : int
        The ID of the spectral window
    num_chan : int
        Expected number of channels in the spectral window

    Returns
    -------
    np.ndarray
        Array of frequency centers for each channel in the spectral window

    Raises
    ------
    RuntimeError
        If the number of frequencies retrieved doesn't match the expected number of channels
    """
    spw_tbl = asdm.getSpectralWindow()
    spw_row = spw_tbl.getRowByKey(pyasdm.types.Tag(f"SpectralWindow_{spw_id}"))
    if spw_row.isChanFreqStartExists():
        freq_start = spw_row.getChanFreqStart().get()
        freq_step = spw_row.getChanFreqStep().get()
        frequency_centers = np.arange(
            freq_start, freq_start + num_chan * freq_step, freq_step
        )
    else:
        frequencies = spw_row.getChanFreqArray()
        frequency_centers = pyasdm.types.Frequency.values(frequencies)
        if len(frequency_centers) != num_chan:
            raise RuntimeError(
                f"Expecting {num_chan} channels but got an array of channel frequencies of length {len(frequency_centers)} "
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

    Notes
    -----
    The function first tries to get a single channel width value. If that doesn't exist,
    it falls back to getting the first value from the channel width array.
    """
    spw_tbl = asdm.getSpectralWindow()
    spw_row = spw_tbl.getRowByKey(pyasdm.types.Tag(f"SpectralWindow_{spw_id}"))
    if spw_row.isChanWidthExists():
        chan_width = spw_row.getChanWidth().get()
    else:
        chan_width = spw_row.getChanWidthArray()[0].get()

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
        ref_frame = spw_row.getMeasFreqRef().get()
    else:
        ref_frame = "TOPO"

    return ref_frame
