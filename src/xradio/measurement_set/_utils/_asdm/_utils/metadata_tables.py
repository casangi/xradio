from typing import Any

import numpy as np
import pandas as pd
import pyasdm

# TODO: check if all cases are covered, liek the the get() also for
# Angle, Length, etc. quantities. Also types: PolarizationType


def load_asdm_col(
    sdm_table: pyasdm.ASDM, col_name: str, optional_per_row: bool = False
) -> list:
    """Load a column from an ASDM table into a list.
    This function extracts values from a specified column in an ASDM (ALMA Science Data Model) table,
    handling various ASDM-specific data types and converting them to Python native types.

    Parameters
    ----------
    sdm_table : pyasdm.ASDM
        An ASDM table object loaded using pyasdm
    col_name : str
        Name of the column to extract from the table
    optional_per_row : bool
        Handle attributes row by row, and do not raise if they are
        absent from some rows.

    Returns
    -------
    list
        List containing the column values converted to appropriate Python types

    Notes
    -----
    Handles special ASDM types including:
    - Tags: converted to integer values
    - EntityRefs: converted to entity IDs
    - Arrays of Tags: converted to numpy arrays
    - Enumerations (StokesParameter, ScanIntent, etc): converted to string names
    - PolarizationType arrays: converted to nested lists of string names
    - Frequency objects: converted to numeric values

    Examples
    --------
    >>> from pyasdm import ASDM
    >>> main_table = ASDM("uid___X02_X1/Main.xml")
    >>> scan_numbers = load_asdm_col(main_table, "scanNumber")
    >>> print(scan_numbers)
    [1, 2, 3, 4]
    """

    def upper_first(col_string):
        return col_string[0].upper() + col_string[1:]

    rows = sdm_table.get()
    get_col_function_name = f"get{upper_first(col_name)}"
    is_col_exists_function_name = f"is{upper_first(col_name)}Exists"
    col_values = []
    for row in rows:
        expect_attribute_present = True
        if optional_per_row:
            has_is_col_exists_function = hasattr(row, is_col_exists_function_name)
            if has_is_col_exists_function:
                is_col_exists_function = getattr(row, is_col_exists_function_name)
                expect_attribute_present = is_col_exists_function()

        # if not has_is_col_exists_function or attribute_present:
        if expect_attribute_present:
            value = _decode_attribute_value(row, get_col_function_name)
        else:
            value = ""

        col_values.append(value)

    return col_values


def _decode_attribute_value(row: Any, get_col_function_name: str) -> Any:
    """ """
    get_col_function = getattr(row, get_col_function_name)
    value = get_col_function()

    # catch Tags and Enumerations
    if isinstance(value, pyasdm.types.Tag):
        # As an int
        value = value.getTagValue()
        # As a string with an int
        # value = value.getTag()
    elif isinstance(value, pyasdm.types.EntityRef):
        value = value.getEntityId()
    elif (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], pyasdm.types.Tag)
    ):
        # Convert the array
        value = np.array([item_val.getTagValue() for item_val in value])
        # if col_name.endswith("Id"):
        #    value = value.astype('int')
    elif (
        isinstance(value, list)
        and len(value) > 0
        and type(value[0])
        in [
            pyasdm.enumerations.StokesParameter,
            pyasdm.enumerations.ScanIntent,
        ]
    ):
        # (or isinstance(value[0], pyasdm.enumerations.PolarizationType))
        # 1-dim array of StokesParameter
        value = [item_val.getName() for item_val in value]
    elif (
        isinstance(value, list)
        and len(value) > 0
        and (
            isinstance(value[0], list)
            and (
                isinstance(value[0][0], pyasdm.enumerations.PolarizationType)
                # or isinstance(value[0][0], pyasdm.enumerations.StokesParameter)
            )
        )
    ):
        # two-dim array of PolarizationType
        value = [
            [
                item_val.getName()
                for first_dim_val in value
                for item_val in first_dim_val
            ]
        ]
        # elif type(value) in pyasdm.enumerations or type(value) in pyasdm.types:
        #    # catch-all
        #    value = value.getValue()
    elif (
        isinstance(value, list)
        and len(value) > 0
        and (
            isinstance(value[0], list)
            and (
                isinstance(value[0][0], pyasdm.types.Angle)
                # or isinstance(value[0][0], pyasdm.enumerations.StokesParameter)
            )
        )
    ):
        # two-dim list/array of list/array of Angle, as seen in Pointing
        value = [
            [item_val.get() for item_val in first_dim_val] for first_dim_val in value
        ]
    elif type(value) in [pyasdm.types.Frequency]:
        value = value.get()
    elif (
        isinstance(value, list)
        and len(value) > 0
        and type(value[0]) in [pyasdm.types.Speed, pyasdm.types.Frequency]
    ):
        # example: Source/sysVel, Source/restFrequency
        value = [item_val.get() for item_val in value]
    elif (
        type(value)
        in [
            pyasdm.enumerations.BasebandName,
            pyasdm.enumerations.ProcessorType,
            pyasdm.enumerations.ProcessorSubType,
            pyasdm.enumerations.SpectralResolutionType,
            pyasdm.enumerations.DirectionReferenceCode,
            pyasdm.enumerations.FrequencyReferenceCode,
        ]
        or ()
    ):
        value = value.getName()

    return value


def exp_asdm_table_to_df(
    sdm: pyasdm.ASDM,
    table_name: str,
    col_names: list[str],
    optional_per_row: bool = False,
) -> pd.DataFrame:
    """Convert an ASDM table to a pandas DataFrame.

    This function extracts specified columns from an ASDM table and converts them into
    a pandas DataFrame format.

    Parameters
    ----------
    sdm : pyasdm.ASDM
        The ASDM object containing the table to be converted
    table_name : str
        Name of the table to extract from the ASDM (without 'Table' suffix)
    col_names : list[str]
        List of column names to extract from the table
    optional_per_row : bool
        Handle attributes row by row, and do not raise if they are
        absent from some rows.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the specified columns from the ASDM table

    Examples
    --------
    >>> df = exp_asdm_table_to_df(sdm, "ExecBlock", ["startTime", "endTime"])
    """

    get_table_name = f"get{table_name}"
    get_table_function = getattr(sdm, get_table_name)
    table = get_table_function()

    col_values = {}
    for col in col_names:
        col_values[col] = load_asdm_col(table, col, optional_per_row)

    df = pd.DataFrame(col_values)
    return df
