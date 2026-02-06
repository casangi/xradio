import numpy as np
import pandas as pd

import pyasdm

# TODO: do the get() also for Angle, Length, etc. quantities
#  Also types: PolarizationType


def load_asdm_col(sdm_table: pyasdm.ASDM, col_name: str) -> list:
    """Load a column from an ASDM table into a list.
    This function extracts values from a specified column in an ASDM (ALMA Science Data Model) table,
    handling various ASDM-specific data types and converting them to Python native types.

    Parameters
    ----------
    sdm_table : pyasdm.ASDM
        An ASDM table object loaded using pyasdm
    col_name : str
        Name of the column to extract from the table

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
    col_values = []
    for row in rows:
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
            # 1-dim array of StoeksParameter
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
        elif type(value) in [pyasdm.types.Frequency]:
            value = value.get()
        elif (
            isinstance(value, list)
            and len(value) > 0
            and type(value[0]) in [pyasdm.types.Speed]
        ):
            # example: Source/sysVel
            value = [item_val.get() for item_val in value]
        elif (
            type(value)
            in [
                pyasdm.enumerations.BasebandName,
                pyasdm.enumerations.ProcessorType,
                pyasdm.enumerations.ProcessorSubType,
                pyasdm.enumerations.SpectralResolutionType,
            ]
            or ()
        ):
            # this branch could/would also include
            # pyasdm.enumerations.FrequencyReferenceCode, etc. enumerations
            value = value.getName()

        col_values.append(value)

    return col_values


def exp_asdm_table_to_df(
    sdm: pyasdm.ASDM, table_name: str, col_names: list[str]
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
        col_values[col] = load_asdm_col(table, col)

    df = pd.DataFrame(col_values)
    return df
