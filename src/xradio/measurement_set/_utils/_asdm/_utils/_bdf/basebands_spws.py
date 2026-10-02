"""
Functions to do various calculations related to the basebands/spw list(s) from
BDF headers.
"""


def calculate_overall_spw_idx(
    basebands_descr: list[dict], baseband_idx: int, spw_idx: int
) -> int:
    overall_spw_idx = (
        sum(
            [
                len(basebands_descr[bb_idx]["spectralWindows"])
                for bb_idx in range(0, baseband_idx)
            ]
        )
        + spw_idx
    )

    return overall_spw_idx


def baseband_spw_to_overall_spw_idx(baseband_spw_idxs, bdf_descr):
    baseband_idx, spw_idx = baseband_spw_idxs
    overall_spw_idx = calculate_overall_spw_idx(
        bdf_descr["basebands"], baseband_idx, spw_idx
    )

    return overall_spw_idx


def find_spw_in_basebands_list(
    spw_id: int,
    basebands: list[dict],
    bdf_path: str,
) -> tuple[int, int]:
    """
    Find the baseband and the SPW within the baseband of an SPW of a BDF.

    The SPWs of a BDF have no IDs, only positions: spw_id is the position of the SPW
    in the list of all the SPWs of the BDF (the SPWs of the first baseband, then the
    SPWs of the second baseband, etc.).

    Parameters
    ----------
    spw_id : int
        Position of the SPW in the BDF (0-based, over all basebands).
    basebands : list[dict]
        Basebands list from the BDF header.
    bdf_path : str
        Path of the BDF (for error messages).

    Returns
    -------
    tuple[int, int]
        Index of the baseband and index of the SPW within that baseband.

    Raises
    ------
    RuntimeError
        If the BDF does not have an SPW at position spw_id (the ASDM metadata and
        the BDF header disagree).
    """
    basebands_len_cumsum = 0
    if spw_id >= 0:
        for baseband_index, bband in enumerate(basebands):
            bb_spw_len = len(bband["spectralWindows"])
            if spw_id < basebands_len_cumsum + bb_spw_len:
                return baseband_index, spw_id - basebands_len_cumsum
            basebands_len_cumsum += bb_spw_len

    raise RuntimeError(
        f"SPW {spw_id} not found in BDF {bdf_path}, which has "
        f"{sum(len(bband['spectralWindows']) for bband in basebands)} SPWs in "
        f"{len(basebands)} basebands. The ASDM metadata (ConfigDescription / "
        "DataDescription) and the BDF header disagree."
    )


def find_if_different_basebands_spws(basebands: list[dict]) -> bool:
    """
    Whether there are different numbers of SPWs in some basebands,
    or different numbers of channels in some SPWs.
    """

    all_same = True
    spws_per_baseband = -1
    chans_per_spw = -1
    for bband in basebands:
        if not all_same:
            break
        num_spws = len(bband["spectralWindows"])
        if spws_per_baseband > 0:
            if num_spws != spws_per_baseband:
                all_same = False
                break
        else:
            spws_per_baseband = num_spws

        for spw in bband["spectralWindows"]:
            num_chans = spw["numSpectralPoint"]
            if chans_per_spw > 0:
                if num_chans != chans_per_spw:
                    all_same = False
                    break
            else:
                chans_per_spw = num_chans

    return not all_same


def find_if_different_basebands_pols(basebands: list[dict]) -> bool:
    """whether the number of polarizations is different for some of the basebands"""

    all_same = True
    spws_per_baseband = -1
    cross_pols_per_spw = -1
    sd_pols_per_spw = -1
    for bband in basebands:
        if not all_same:
            break
        num_spws = len(bband["spectralWindows"])
        if spws_per_baseband > 0:
            if num_spws != spws_per_baseband:
                all_same = False
                break
        else:
            spws_per_baseband = num_spws

        for spw in bband["spectralWindows"]:
            num_pols_cross = len(spw["crossPolProducts"])
            num_pols_sd = len(spw["sdPolProducts"])
            if cross_pols_per_spw > 0:
                if num_pols_cross != cross_pols_per_spw:
                    all_same = False
                    break
            else:
                cross_pols_per_spw = num_pols_cross
            if sd_pols_per_spw > 0:
                if num_pols_sd != sd_pols_per_spw:
                    all_same = False
                    break
            else:
                sd_pols_per_spw = num_pols_sd

    return not all_same
