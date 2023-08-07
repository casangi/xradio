#  CASA Next Generation Infrastructure
#  Copyright (C) 2023 AUI, Inc. Washington DC, USA
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Tuple

from casacore import tables
import numpy as np


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

    :param tb_tool: a table/TaQL query open and being used to load columns
    :param col: colum to load
    :param cshape: shape of the resulting col data chunk
    :param tidxs: time axis indices
    :param bidxs: baseline axis indices
    :param didxs: (effective) data indices, excluding missing baselines
    :param d1: indices to load on dimension 1 (None=all, pols or chans)
    :param d2: indices to load on dimension 2 (None=all, pols)

    :return: data array loaded directly with casacore getcol/getcolslice
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
    policy = "warn"
    if np.issubdtype(data.dtype, np.integer):
        policy = "ignore"
    with np.errstate(invalid=policy):
        chunk = np.full(cshape, np.nan, dtype=data.dtype)
    if len(didxs) > 0:
        chunk[tidxs[didxs], bidxs[didxs]] = data[didxs]

    return chunk
