from typing import Any

# ASDM DirectionReferenceCode -> astropy SkyCoord frame names, as used in the
# MSv4 "frame" attribute of sky coordinates. Follows the translations applied by
# the MSv2 converter to casacore frames (J2000 => fk5, ICRS => icrs, AZELGEO =>
# altaz), extended with the remaining codes that have an unambiguous astropy
# equivalent. Codes not in this map (JMEAN, JTRUE, APP, B1950_VLA, BMEAN, BTRUE,
# AZELSW*, JNAT, *ECLIPTIC, TOPO, planets, ...) are not supported.
# This map only labels directions. Interferometric phase centres must also be
# in a frame supported by the UVW calculation (calculate_uvw): see
# create_field_and_source_xds, which rejects the other frames (hadec, altaz,
# itrs, which would need an observation time and location) when a partition is
# opened.
ASDM_DIRECTION_CODE_TO_FRAME = {
    "J2000": "fk5",
    "ICRS": "icrs",
    "B1950": "fk4",
    "GALACTIC": "galactic",
    "SUPERGAL": "supergalactic",
    "HADEC": "hadec",
    # Azimuth measured from North through East, as astropy AltAz (the AZELSW*
    # codes count azimuth from South through West and have no astropy frame).
    "AZELNE": "altaz",
    "AZELNEGEO": "altaz",
    "ITRF": "itrs",
}

# Direction code assumed when the optional ASDM directionCode is absent.
ASDM_DEFAULT_DIRECTION_CODE = "J2000"

# ASDM FrequencyReferenceCode -> MSv4 spectral_coord "observer", following the
# translations applied by the MSv2 converter to casacore frequency frames.
ASDM_FREQUENCY_CODE_TO_OBSERVER = {
    "REST": "REST",
    "LABREST": "REST",
    "LSRK": "lsrk",
    "LSRD": "lsrd",
    "BARY": "BARY",
    "GEO": "gcrs",
    "TOPO": "TOPO",
}


def make_field_name(field_name: str, field_id: int) -> str:
    """
    Build the MSv4 field name for an ASDM field.

    ASDM field names are not unique (for example, all the pointings of an
    ALMA mosaic usually share one name), so the field id is appended, as the
    MSv2 converter does: ``f"{field_name}_{field_id}"``.

    Parameters
    ----------
    field_name : str
        Field name from the ASDM Field table (``Field.fieldName``).
    field_id : int
        Field id (the integer value of the ``Field.fieldId`` tag).

    Returns
    -------
    str
        Unique field name.
    """
    return f"{str(field_name).strip()}_{int(field_id)}"


def get_optional_row_attr(row: Any, attr_name: str, default: Any = None) -> Any:
    """
    Get an optional attribute of a pyasdm table row.

    pyasdm raises ``ValueError`` when the getter of an absent optional
    attribute is called, so the ``is<Attr>Exists`` method is checked first.

    Parameters
    ----------
    row : Any
        pyasdm table row (for example a ``FieldRow`` or ``SourceRow``).
    attr_name : str
        Attribute name as in the ASDM table (for example ``"directionCode"``).
    default : Any, optional
        Value returned when the attribute is absent.

    Returns
    -------
    Any
        The attribute value (as returned by the pyasdm getter), or ``default``.
    """
    upper_name = attr_name[0].upper() + attr_name[1:]
    if getattr(row, f"is{upper_name}Exists")():
        return getattr(row, f"get{upper_name}")()
    return default


def direction_code_name(direction_code: Any) -> str:
    """
    Name of an ASDM direction reference code.

    Parameters
    ----------
    direction_code : Any
        ASDM ``DirectionReferenceCode`` enumeration value or its name, or ``None``
        (absent, which gives the default ``J2000``).

    Returns
    -------
    str
        Upper case code name, for example ``"ICRS"``.
    """
    if direction_code is None:
        code_name = ASDM_DEFAULT_DIRECTION_CODE
    elif hasattr(direction_code, "getName"):
        code_name = direction_code.getName()
    else:
        code_name = str(direction_code)
    return code_name.strip().upper()


def direction_code_to_frame(direction_code: Any) -> str:
    """
    Translate an ASDM direction reference code into an astropy frame name.

    Parameters
    ----------
    direction_code : Any
        ASDM ``DirectionReferenceCode`` enumeration value or its name (for example
        ``"ICRS"`` or ``"J2000"``). ``None`` means absent and gives the default
        (``J2000``, i.e. ``fk5``).

    Returns
    -------
    str
        Lowercase astropy frame name as used in the MSv4 ``frame`` attribute.

    Raises
    ------
    ValueError
        If the direction code has no supported astropy equivalent.
    """
    code_name = direction_code_name(direction_code)
    try:
        return ASDM_DIRECTION_CODE_TO_FRAME[code_name]
    except KeyError:
        raise ValueError(
            f"The ASDM direction reference code {code_name} is not supported. "
            f"Supported codes: {sorted(ASDM_DIRECTION_CODE_TO_FRAME)}."
        ) from None


def frequency_code_to_observer(frequency_code: Any, default: str = "lsrk") -> str:
    """
    Translate an ASDM frequency reference code into an MSv4 spectral observer.

    Parameters
    ----------
    frequency_code : Any
        ASDM ``FrequencyReferenceCode`` enumeration value or its name, or ``None``
        (absent).
    default : str, optional
        Observer returned when the code is absent, by default ``"lsrk"`` (the frame
        of the rest frequencies of datasets converted from MSv2).

    Returns
    -------
    str
        Spectral observer name as used in the MSv4 ``observer`` attribute.

    Raises
    ------
    ValueError
        If the frequency code has no MSv4 equivalent.
    """
    if frequency_code is None:
        return default
    if hasattr(frequency_code, "getName"):
        code_name = frequency_code.getName()
    else:
        code_name = str(frequency_code)
    code_name = code_name.strip().upper()

    try:
        return ASDM_FREQUENCY_CODE_TO_OBSERVER[code_name]
    except KeyError:
        raise ValueError(
            f"The ASDM frequency reference code {code_name} is not supported. "
            f"Supported codes: {sorted(ASDM_FREQUENCY_CODE_TO_OBSERVER)}."
        ) from None
