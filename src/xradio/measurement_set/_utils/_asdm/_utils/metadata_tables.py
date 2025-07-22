import numpy as np
import pandas as pd

import pyasdm


# TODO: do the get() also for Angle, Length, etc. quantities
def load_asdm_col(sdm_table: pyasdm.ASDM, col_name: str) -> list:
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
        elif isinstance(value, list) and isinstance(value[0], pyasdm.types.Tag):
            # Convert the array
            value = np.array([val.getTagValue() for val in value])
            # if col_name.endswith("Id"):
            #    value = value.astype('int')
        elif isinstance(value, list) and type(value[0]) in [
            pyasdm.enumerations.StokesParameter,
            pyasdm.enumerations.ScanIntent,
        ]:
            # (or isinstance(value[0], pyasdm.enumerations.PolarizationType))
            # 1-dim array of StoeksParameter
            value = [val.getName() for val in value]
        elif isinstance(value, list) and (
            isinstance(value[0], list)
            and (
                isinstance(value[0][0], pyasdm.enumerations.PolarizationType)
                # or isinstance(value[0][0], pyasdm.enumerations.StokesParameter)
            )
        ):
            # two-dim array of PolarizationType
            value = [
                [val.getName() for first_dim_val in value for val in first_dim_val]
            ]
            # elif type(value) in pyasdm.enumerations or type(value) in pyasdm.types:
            #    # catch-all
            #    value = value.getValue()
        elif type(value) in [pyasdm.types.Frequency]:
            value = value.get()
        elif (
            type(value)
            in [
                pyasdm.enumerations.BasebandName,
                pyasdm.enumerations.ProcessorType,
                pyasdm.enumerations.ProcessorSubType,
            ]
            or ()
        ):
            value = value.getName()

        col_values.append(value)

    return col_values


def exp_asdm_table_to_df(
    sdm: pyasdm.ASDM, table_name: str, col_names: list[str]
) -> pd.DataFrame:
    get_table_name = f"get{table_name}"
    get_table_function = getattr(sdm, get_table_name)
    table = get_table_function()

    col_values = {}
    for col in col_names:
        col_values[col] = load_asdm_col(table, col)

    df = pd.DataFrame(col_values)
    return df
