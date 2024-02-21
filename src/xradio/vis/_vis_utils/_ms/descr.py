import os
from typing import Dict, Union

import numpy as np
import pandas as pd

from ._tables.read import read_generic_table, read_flat_col_chunk
from ._tables.table_query import open_query, open_table_ro
from xradio.vis._vis_utils._ms.optimised_functions import unique_1d


def describe_ms(
    infile: str, mode="summary", rowmap: Union[dict, None] = None
) -> Union[pd.DataFrame, Dict]:
    """
    Summarize the contents of an MS directory in casacore table format

    :param infile: input MS filename
    :param mode: type of information returned ('summary', 'flat', 'expanded').
    'summary' returns a pandas dataframe that is nice for displaying in notebooks
    etc. 'flat' returns a list of tuples of (ddi, row, chan, pol). 'expanded'
    returns a list of tuples of (ddi, time, baseline, chan, pol). These latter two
    are good for trying to determine chunk size for read_ms(expand=True/False).
    :param rowmap: dict of DDI to tuple of (row indices, channel indices). Returned
    by ms_selection function. Default None ignores selections

    :return: summary as a pd dataframe
    """
    infile = os.path.expanduser(infile)  # does nothing if $HOME is unknown
    if not os.path.isdir(infile):
        raise ValueError(f"invalid input filename to describe_ms: {infile}")
    if mode not in [
        "summary",
        "flat",
        "expanded",
    ]:
        raise ValueError("invalid mode, must be summary, flat or expanded")

    ddi_xds = read_generic_table(infile, "DATA_DESCRIPTION")
    ddis = list(ddi_xds.row.values) if rowmap is None else list(rowmap.keys())

    summary: Union[pd.DataFrame, Dict] = []
    if mode == "summary":
        summary = pd.DataFrame([])

    with open_table_ro(infile) as tb_tool:
        for ddi in ddis:
            taql = f"select * from $tb_tool where DATA_DESC_ID = {ddi}"
            with open_query(tb_tool, taql) as query_per_ddi:
                populate_ms_descr(infile, mode, query_per_ddi, summary, ddi, ddi_xds)

    if mode == "summary":
        summary = summary.set_index("ddi").sort_index()
    else:
        summary = dict(summary)
    return summary


def populate_ms_descr(
    infile, mode, query_per_ddi, summary, ddi, ddi_xds, rowmap=None
) -> pd.DataFrame:
    """Adds information from the time and baseline (antenna1+antenna2)
    columns as well as channel and polarizations, based on a taql
    query.

    :param mode: mode (as in describe_ms())
    :param query_per_ddi: a TaQL query with data per individual DDI
    :param sdf: summary dict being populated
    :param summary: final summary object being populated from the invividual sdf's
    """
    spw_ids = ddi_xds.spectral_window_id.values
    pol_ids = ddi_xds.polarization_id.values
    sdf = {
        "ddi": ddi,
        "spw_id": spw_ids[ddi],
        "pol_id": pol_ids[ddi],
        "rows": query_per_ddi.nrows(),
    }

    # figure out characteristics of main table from select subtables (must all be present)
    spw_xds = read_generic_table(infile, "SPECTRAL_WINDOW")
    pol_xds = read_generic_table(infile, "POLARIZATION")

    if mode in ["expanded", "summary"]:
        times = (
            query_per_ddi.getcol("TIME")
            if rowmap is None
            else read_flat_col_chunk(infile, "TIME", (1,), rowmap[ddi][0], 0, 0)
        )
        baselines = [
            (
                query_per_ddi.getcol(rr)[:, None]
                if rowmap is None
                else read_flat_col_chunk(infile, rr, (1,), rowmap[ddi][0], 0, 0)
            )
            for rr in ["ANTENNA1", "ANTENNA2"]
        ]
        sdf.update(
            {
                "times": len(unique_1d(times)),
                "baselines": len(np.unique(np.hstack(baselines), axis=0)),
            }
        )

    chans = spw_xds.num_chan.values
    pols = pol_xds.num_corr.values
    sdf.update(
        {
            "chans": (
                chans[spw_ids[ddi]]
                if (rowmap is None) or (rowmap[ddi][1] is None)
                else len(rowmap[ddi][1])
            ),
            "pols": pols[pol_ids[ddi]],
        }
    )
    sdf["size_MB"] = np.ceil(
        (sdf["rows"] * sdf["chans"] * sdf["pols"] * 10) / 1024**2
    ).astype(int)

    if rowmap is not None:
        sdf["rows"] = len(rowmap[ddi][0])

    if mode == "summary":
        summary = pd.concat(
            [summary, pd.DataFrame(sdf, index=[str(ddi)])], axis=0, sort=False
        )
    elif mode == "flat":
        summary += [(ddi, (sdf["rows"], sdf["chans"], sdf["pols"]))]
    else:
        summary += [((ddi, sdf["times"], sdf["baselines"], sdf["chans"], sdf["pols"]))]

    return sdf
