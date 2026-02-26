"""
Functions to do various calculations related to the basebands/spw list(s).
"""

import toolviper.utils.logger as logger


def calculate_overall_spw_idx(
    basebands_descr: list[dict], baseband_idx: int, spw_idx: int
) -> int:
    overall_spw_idx = sum(
        [
            len(basebands_descr[bb_idx]["spectralWindows"])
            for bb_idx in range(0, baseband_idx)
        ]
    )
    +spw_idx

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

    bb_index_cnt = 0
    basebands_len_cumsum = 0
    found = False
    for bband in basebands:
        bb_spw_len = len(bband["spectralWindows"])
        if spw_id < basebands_len_cumsum + bb_spw_len:
            spw_index = spw_id - basebands_len_cumsum
            baseband_index = bb_index_cnt
            found = True
            break
        else:
            basebands_len_cumsum += bb_spw_len

        bb_index_cnt += 1

    if not found:
        # TODO: This is a highly dubious fallback for now...
        # raise RuntimeError(err_msg)
        err_msg = f"SPW {spw_id} not found in this BDF: {bdf_path}, defaulting to BB 0, SPW 0."
        logger.warning(err_msg)
        spw_index = 1 - 1
        baseband_index = 0

    return (baseband_index, spw_index)
