import pandas as pd
import pyasdm

from xradio._utils.schema import casacore_to_msv4_measure_type


def retrieve_field_direction_frame(field_df: pd.DataFrame, field_id: int) -> str:
    """Return the direction reference frame for a field.

    Parameters
    ----------
    field_df : pandas.DataFrame
        Field table containing ``fieldId`` and ``directionCode`` columns.
    field_id : int
        Identifier of the field to inspect.

    Returns
    -------
    str
        The mapped frame name, or ``"fk5"`` if the direction code is missing
        or is NaN. An unmapped code is returned in lowercase.
    """
    try:
        direction_code = field_df.loc[field_df["fieldId"] == field_id][
            "directionCode"
        ].item()
    except (AttributeError, ValueError) as _exc:
        direction_code = ""

    # directionCode doc: "The common reference code for directions in the row, if not J2000."
    if direction_code and not pd.isna(direction_code):
        # Seen so far: ICRS, J2000, which are in this map:
        direction_frame = casacore_to_msv4_measure_type["direction"]["Ref_map"].get(
            direction_code, direction_code.lower()
        )
    else:
        direction_frame = "fk5"

    return direction_frame


def get_source_direction_code(
    asdm: pyasdm.ASDM, source_key: tuple[int, object, int]
) -> str:
    """Get the direction frame code for a source in an ASDM dataset.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM dataset object.
    source_key : tuple
        A tuple ``(source_id, time_interval, spectral_window_id)`` identifying
        a source. The time interval is passed through to the source-table lookup.

    Returns
    -------
    str
        The lowercase direction code. ``"j2000"`` is normalized to ``"fk5"``;
        ``"fk5"`` is also returned if the source has no direction code.

    Notes
    -----
    The direction code specifies the reference frame of the source coordinates
    (for example, ``"fk5"`` or ``"icrs"``).
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


def retrieve_source_frequency_frame(source_df: pd.DataFrame) -> str:
    """Return the frequency reference frame for a source.

    Parameters
    ----------
    source_df : pandas.DataFrame
        Source table row containing a ``frequencyRefCode`` column.

    Returns
    -------
    str
        The mapped frame name, or ``"lsrk"`` if the reference code is missing
        or is NaN. An unmapped code is returned in lowercase.
    """
    frequency_ref_code = source_df["frequencyRefCode"].item()

    # frequencyRefCode doc: "The reference code for frequency (if not LSRK)"
    if frequency_ref_code and not pd.isna(frequency_ref_code):
        frequency_frame = casacore_to_msv4_measure_type["frequency"]["Ref_map"].get(
            frequency_ref_code, frequency_ref_code.lower()
        )
    else:
        frequency_frame = "lsrk"

    return frequency_frame
