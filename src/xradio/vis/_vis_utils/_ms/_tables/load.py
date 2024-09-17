from typing import Tuple

import numpy as np

from casacore import tables
from ....._utils.common import get_pad_value


def load_col_chunk(
    tb_tool: tables.table,
    col: str,
    cshape: Tuple[int],
    tidxs: np.ndarray,
    bidxs: np.ndarray,
    didxs: np.ndarray,
    d1: Tuple[int, int],
    d2: Tuple[int, int],
) -> np.ndarray:
    """
    Loads a slice of a col (using casacore getcol(slice))

    Parameters
    ----------
    tb_tool : tables.table
        a table/TaQL query open and being used to load columns
    col : str
        colum to load
    cshape : Tuple[int]
        shape of the resulting col data chunk
    tidxs : np.ndarray
        time axis indices
    bidxs : np.ndarray
        baseline axis indices
    didxs : np.ndarray
        effective) data indices, excluding missing baselines
    d1 : Tuple[int, int]
        indices to load on dimension 1 (None=all, pols or chans)
    d2 : Tuple[int, int]
        indices to load on dimension 2 (None=all, pols)

    Returns
    -------
    np.ndarray
        data array loaded directly with casacore getcol/getcolslice
    """

    if (len(cshape) == 2) or (col == "UVW"):
        # all the scalars and UVW
        data = np.array(tb_tool.getcol(col, 0, -1))
    elif len(cshape) == 3:
        # WEIGHT, SIGMA
        data = tb_tool.getcolslice(col, d1[0], d1[1], [], 0, -1)
    elif len(cshape) == 4:
        # DATA, FLAG, WEIGHT_SPECTRUM / SIGMA_SPECTRUM
        data = tb_tool.getcolslice(col, (d1[0], d2[0]), (d1[1], d2[1]), [], 0, -1)

    # full data is the maximum of the data shape and chunk shape dimensions
    fill_value = get_pad_value(data.dtype)
    chunk = np.full(cshape, np.fill_value, dtype=data.dtype)
    if len(didxs) > 0:
        chunk[tidxs[didxs], bidxs[didxs]] = data[didxs]

    return chunk
