"""
Based on a BDF description (metadata from the BDF header), calculates the offsets for the
flags of a given SPW. These offsets are the position increments (increments 'before' and
'after') where the flags for that SPW can be found in the flags binary component 1d array.
"""

import numpy as np

import pyasdm


def calculate_offset_additions_cross_sd(
    bdf_descr: list[dict],
    baseband_idx: int,
    overall_spw_idx: int,
    flag_array_len: int,
) -> dict[str, tuple[int, int]]:

    auto_only = (
        bdf_descr["correlation_mode"] == pyasdm.enumerations.CorrelationMode.AUTO_ONLY
    )

    basebands_descr = bdf_descr["basebands"]
    antenna_len = bdf_descr["num_antenna"]
    baseline_len = int(antenna_len * (antenna_len - 1) / 2)

    per_spw_cross_pol_lens = _make_per_spw_cross_pol_lens(basebands_descr)
    if auto_only:
        guessed_cross_len = 0
    else:
        guessed_cross_len = baseline_len * np.sum(per_spw_cross_pol_lens, dtype=int)

    per_spw_auto_pol_lens = _make_per_spw_auto_pol_lens(basebands_descr)
    guessed_auto_len = antenna_len * np.sum(per_spw_auto_pol_lens, dtype=int)
    time_len = bdf_descr["num_time"] if bdf_descr["dimensionality"] == 0 else 1

    offset = {"cross": {}, "auto": {}}
    if (guessed_cross_len + guessed_auto_len) * time_len == flag_array_len:
        offset = _make_flag_tree_offsets_per_spw(
            per_spw_cross_pol_lens, per_spw_auto_pol_lens, overall_spw_idx
        )
    else:
        # Try per-BB flags
        per_baseband_cross_pol_lens = _make_per_baseband_cross_pol_lens(basebands_descr)
        if auto_only:
            second_guessed_cross_len = 0
        else:
            second_guessed_cross_len = baseline_len * np.sum(
                per_baseband_cross_pol_lens, dtype=int
            )

        per_baseband_auto_pol_lens = _make_per_baseband_auto_pol_lens(basebands_descr)
        second_guessed_auto_len = antenna_len * np.sum(
            per_baseband_auto_pol_lens, dtype=int
        )

        if (
            second_guessed_cross_len + second_guessed_auto_len
        ) * time_len == flag_array_len:
            offset = _make_flag_tree_offsets_per_baseband(
                per_baseband_cross_pol_lens, per_baseband_auto_pol_lens, baseband_idx
            )
        else:
            raise RuntimeError(
                f"Unexpected flags array in a subset. {guessed_cross_len=}, {guessed_auto_len=}, "
                f"{second_guessed_cross_len=}, {second_guessed_auto_len=}, {flag_array_len=} "
                f"{offset=}, {bdf_descr=}"
            )

    return offset


def _make_per_spw_cross_pol_lens(basebands_descr: list[list]) -> int:
    per_spw_cross_pol_lens = [
        len(basebands_descr[bb_idx]["spectralWindows"][spw_idx]["crossPolProducts"])
        for bb_idx in range(0, len(basebands_descr))
        for spw_idx in range(0, len(basebands_descr[bb_idx]["spectralWindows"]))
    ]
    return per_spw_cross_pol_lens


def _make_per_spw_auto_pol_lens(basebands_descr: list[list]) -> int:
    per_spw_auto_pol_len = [
        len(basebands_descr[bb_idx]["spectralWindows"][spw_idx]["sdPolProducts"])
        for bb_idx in range(0, len(basebands_descr))
        for spw_idx in range(0, len(basebands_descr[bb_idx]["spectralWindows"]))
    ]
    return per_spw_auto_pol_len


def _make_per_baseband_cross_pol_lens(basebands_descr: list[list]) -> int:
    per_baseband_cross_pol_lens = [
        len(basebands_descr[bb_idx]["spectralWindows"][0]["crossPolProducts"])
        for bb_idx in range(0, len(basebands_descr))
    ]
    return per_baseband_cross_pol_lens


def _make_per_baseband_auto_pol_lens(basebands_descr: list[list]) -> int:
    per_baseband_auto_pol_lens = [
        len(basebands_descr[bb_idx]["spectralWindows"][0]["sdPolProducts"])
        for bb_idx in range(0, len(basebands_descr))
    ]
    return per_baseband_auto_pol_lens


def _make_flag_tree_offsets_per_spw(
    per_spw_cross_pol_lens: list[int],
    per_spw_auto_pol_lens: list[int],
    overall_spw_idx: int,
) -> dict[str, dict]:

    offset = {"cross": {}, "auto": {}}
    offset["cross"]["before"] = np.sum(
        per_spw_cross_pol_lens[0:overall_spw_idx], dtype=int
    )
    offset["cross"]["after"] = np.sum(
        per_spw_cross_pol_lens[overall_spw_idx:], dtype=int
    )

    offset["auto"]["before"] = np.sum(
        per_spw_auto_pol_lens[0:overall_spw_idx], dtype=int
    )
    offset["auto"]["after"] = np.sum(per_spw_auto_pol_lens[overall_spw_idx:], dtype=int)

    return offset


def _make_flag_tree_offsets_per_baseband(
    per_baseband_cross_pol_lens, per_baseband_auto_pol_lens, baseband_idx
) -> dict[str, dict]:

    offset = {"cross": {}, "auto": {}}
    offset["cross"]["before"] = np.sum(
        per_baseband_cross_pol_lens[0:baseband_idx], dtype=int
    )
    offset["cross"]["after"] = np.sum(
        per_baseband_cross_pol_lens[baseband_idx:], dtype=int
    )

    offset["auto"]["before"] = np.sum(
        per_baseband_auto_pol_lens[0:baseband_idx], dtype=int
    )
    offset["auto"]["after"] = np.sum(
        per_baseband_auto_pol_lens[baseband_idx:], dtype=int
    )

    return offset
