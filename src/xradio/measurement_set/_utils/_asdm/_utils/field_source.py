import pyasdm


def get_direction_codes(asdm: pyasdm.ASDM, source_key: tuple) -> "str":
    """
    Get the direction code from a source in an ASDM dataset.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM dataset object.
    source_key : tuple
        A tuple containing (source_id, time_interval, spectral_window_id) that identifies
        a specific source in the ASDM.

    Returns
    -------
    str
        The direction code of the source. Returns 'fk5' if no direction code exists.

    Notes
    -----
    The direction code specifies the reference frame of the source coordinates
    (e.g., 'fk5', 'icrs', etc.).
    """

    source_id = int(source_key[0])
    # ignoring interval:
    # time_interval = source_key[1]
    spectral_window_id = int(source_key[2])
    key = (
        source_id,
        source_key[1],
        pyasdm.types.Tag(f"SpectralWindow_{spectral_window_id}"),
    )

    source_tbl = asdm.getSource()
    source_row = source_tbl.getRowByKey(*key)
    if source_row.isDirectionCodeExists():
        direction_code = source_row.getDirectionCode().getName().lower()
        if direction_code == "j2000":
            direction_code = "fk5"
    else:
        direction_code = "fk5"

    return direction_code
