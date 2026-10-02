"""
Loads the visibilities/flags of one SPW from the binary components of the subsets of
a BDF, looking for the data specific to that SPW in the data trees of the binary
components (and skipping the 'other' SPWs).

Visibilities are loaded either with pyasdm BDFReader.getNDArrays(), which reads only
the blocks needed for the SPW directly from the file (see
pyasdm_get_ndarray_load_function), or from the 'arr' 1D arrays produced by
pyasdm BDFReader.getSubset() (all the SPWs loaded, then the SPW selected). Flags are
loaded from the 'arr' arrays of getSubset(). This procedure works for any BDF,
regardless of the configuration of basebands and SPWs (numbers of SPWs per baseband,
channels per SPW, polarizations per SPW).

The time dimension counts integrations: one per subset, or numTime per subset for
packed BDFs (dimensionality 0). Subsets outside the selected time range are skipped
without loading their data, and reading stops after the last selected subset.

The layout of the SPW in the BDF is calculated once per BDF load and passed to the
per-subset (and per-binary-component) functions. The data of every subset are
written straight into their place in the result (or in an ``out`` array given by
the caller), without per-subset arrays.
"""

from collections.abc import Callable, Iterator

import numpy as np
import pyasdm

from xradio.measurement_set._utils._asdm._utils._bdf import config
from xradio.measurement_set._utils._asdm._utils._bdf.basebands_spws import (
    baseband_spw_to_overall_spw_idx,
    calculate_overall_spw_idx,
)
from xradio.measurement_set._utils._asdm._utils._bdf.flags_offsets import (
    calculate_offset_additions_cross_sd,
)
from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_get_ndarray_load_function import (
    VISIBILITY_DTYPE,
    load_auto_data_one_spw,
    load_cross_data_one_spw,
    load_visibilities_one_spw_to_ndarray,
    output_array,
    read_component_block,
    selection_shape,
)
from xradio.measurement_set._utils._asdm._utils._bdf.shapes import (
    BDFSelection,
    BDFSpwLayout,
    calc_bdf_spw_layout,
    normalize_bdf_selection,
)

#: Components loaded when skipping a subset. The (small) actualTimes component is
#: given because an empty selection makes pyasdm load every binary component.
SKIP_SUBSET_COMPONENTS = {"actualTimes"}
#: dtype of the flags produced by the loaders
FLAG_DTYPE = np.dtype(bool)


def load_visibilities_all_subsets_from_trees(
    bdf_reader: pyasdm.bdf.BDFReader,
    guessed_shape: tuple[int, ...],
    baseband_spw_idxs: tuple[int, int],
    bdf_descr: dict,
    array_slice: tuple[slice, ...],
    load_one_spw_from_file: bool | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Loads the visibilities of one SPW from the subsets of a BDF.

    Parameters
    ----------
    bdf_reader : pyasdm.bdf.BDFReader
        Reader of the BDF, opened and positioned before the first subset.
    guessed_shape : tuple[int, ...]
        Unused (the layout of the data is derived from bdf_descr). Kept for
        compatibility.
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband and index of the SPW within the baseband.
    bdf_descr : dict
        BDF description (from the BDF header).
    array_slice : tuple[slice, ...]
        Selection (time, baseline, frequency, polarization). The time slice counts
        the integrations of the BDF (TIM samples for packed BDFs). The baseline
        slice applies to the cross baselines followed by the autos (or to the
        antennas for AUTO_ONLY data). Slices must have step 1; int keys select one
        index and keep the dimension.
    load_one_spw_from_file : bool | None
        Load with BDFReader.getNDArrays() reading only the needed blocks from the
        file, or with BDFReader.getSubset(). None (default): the value of
        config.use_load_one_spw_at_a_time when the function is called.
    out : np.ndarray | None
        complex64 array with the shape of the selection to write the visibilities
        into (for example a view of a larger array). It also gives the length of
        the time selection when the time slice has no stop. None: a new array.

    Returns
    -------
    np.ndarray
        complex64 array (time, baseline, frequency, polarization): out, if given.
    """
    if load_one_spw_from_file is None:
        load_one_spw_from_file = config.use_load_one_spw_at_a_time
    layout = calc_bdf_spw_layout(bdf_descr, baseband_spw_idxs)
    selection = normalize_bdf_selection(array_slice, layout)
    components_to_load = _find_data_components_to_load(selection, layout)
    overall_spw_idx = baseband_spw_to_overall_spw_idx(baseband_spw_idxs, bdf_descr)

    def load_subset_block(subset_slice: tuple[slice, ...], block: np.ndarray):
        if load_one_spw_from_file:
            loaded_components = set()
            load_subset_with_get_ndarrays(
                bdf_reader,
                overall_spw_idx,
                load_visibilities_one_spw_to_ndarray,
                (
                    bdf_descr,
                    components_to_load,
                    guessed_shape,
                    subset_slice,
                    block,
                    layout,
                    loaded_components,
                ),
            )
            missing = [
                name for name in components_to_load if name not in loaded_components
            ]
            if missing:
                raise RuntimeError(
                    f"Binary component(s) {missing} not present in the BDF subset."
                )
        else:
            subset = load_subset_with_get_subset(bdf_reader, components_to_load)
            load_vis_subset_from_tree(
                subset,
                guessed_shape,
                baseband_spw_idxs,
                bdf_descr,
                subset_slice,
                out=block,
                layout=layout,
            )

    return _load_selected_subsets(
        bdf_reader,
        layout,
        selection,
        load_subset_block,
        (_range_len(selection.frequency), _range_len(selection.polarization)),
        VISIBILITY_DTYPE,
        out,
    )


def _find_data_components_to_load(
    selection: BDFSelection, layout: BDFSpwLayout
) -> list[str]:
    components = []
    if selection.cross_range(layout.num_cross_baselines) is not None:
        components.append("crossData")
    if selection.auto_range(layout.num_cross_baselines) is not None:
        components.append("autoData")
    return components


def _range_len(index_range: tuple[int, int]) -> int:
    return index_range[1] - index_range[0]


def skip_subset(bdf_reader: pyasdm.bdf.BDFReader):
    """
    Moves the reader past the next subset, without loading its data and flags.

    Parameters
    ----------
    bdf_reader : pyasdm.bdf.BDFReader
        Reader of the BDF.
    """
    load_subset_with_get_subset(bdf_reader, SKIP_SUBSET_COMPONENTS)


def _iterate_selected_subsets(
    bdf_reader: pyasdm.bdf.BDFReader,
    times_per_subset: int,
    time_range: tuple[int, int | None],
) -> Iterator[tuple[int, int, int]]:
    """
    Iterates through the subsets that have integrations in time_range. The other
    subsets are skipped, and no subset is read after the last one selected. The
    caller must read every subset yielded (with getSubset() or getNDArrays()).

    Yields (first integration of the subset, start, stop), with start/stop relative
    to the integrations of the subset.
    """
    time_start, time_stop = time_range
    subset_start = 0
    any_selected = False
    while (time_stop is None or subset_start < time_stop) and bdf_reader.hasSubset():
        subset_stop = subset_start + times_per_subset
        if subset_stop <= time_start:
            skip_subset(bdf_reader)
        else:
            any_selected = True
            stop = subset_stop if time_stop is None else min(time_stop, subset_stop)
            yield (
                subset_start,
                max(time_start - subset_start, 0),
                stop - subset_start,
            )
        subset_start = subset_stop

    if (time_stop is not None and subset_start < time_stop) or not any_selected:
        raise ValueError(
            f"Time indices [{time_start}, {time_stop}) requested, but the BDF has "
            f"only {subset_start} integrations."
        )


def _load_selected_subsets(
    bdf_reader: pyasdm.bdf.BDFReader,
    layout: BDFSpwLayout,
    selection: BDFSelection,
    load_subset_block: Callable,
    trailing_shape: tuple[int, ...],
    dtype: np.dtype,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Loads the selection from the subsets of a BDF, calling
    load_subset_block(subset_slice, block) for every subset with integrations in
    the time range, with the selection relative to the subset and the array where
    the data of the subset must be written (its part of the result).

    The result (out, if given) is allocated once and every subset writes into its
    part of it. Only when the length of the time selection is unknown (time slice
    without stop and no out) are the subsets loaded into separate arrays, then
    assembled (no copy when there is only one).
    """
    time_start, time_stop = selection.time
    baseline_start, baseline_stop = selection.baseline
    other_slices = (
        slice(baseline_start, baseline_stop),
        slice(*selection.frequency),
        slice(*selection.polarization),
    )
    block_shape = (baseline_stop - baseline_start, *trailing_shape)

    if out is not None and time_stop is None:
        if not isinstance(out, np.ndarray) or out.ndim != len(block_shape) + 1:
            raise ValueError(
                f"Output array with shape {getattr(out, 'shape', None)}, expected "
                f"(time, *{block_shape})"
            )
        time_stop = time_start + out.shape[0]
    if time_stop is not None:
        result = output_array(out, (time_stop - time_start, *block_shape), dtype)
    else:
        result = None
        blocks = []

    for subset_start, start, stop in _iterate_selected_subsets(
        bdf_reader, layout.times_per_subset, (time_start, time_stop)
    ):
        if result is not None:
            offset = subset_start + start - time_start
            block = result[offset : offset + stop - start]
        else:
            block = np.empty((stop - start, *block_shape), dtype=dtype)
            blocks.append(block)
        load_subset_block((slice(start, stop), *other_slices), block)

    if result is None:
        result = blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    return result


def load_subset_with_get_subset(
    bdf_reader: pyasdm.bdf.BDFReader, components_to_load: list[str] | set[str]
) -> dict:
    """
    Reads the next subset with BDFReader.getSubset(), loading only some binary
    components.

    Parameters
    ----------
    bdf_reader : pyasdm.bdf.BDFReader
        Reader of the BDF.
    components_to_load : list[str] | set[str]
        Binary components to load (must not be empty: pyasdm would load all the
        components).

    Returns
    -------
    dict
        The subset, as returned by BDFReader.getSubset().
    """
    if not components_to_load:
        raise ValueError("No binary component to load from the BDF subset")

    try:
        subset = bdf_reader.getSubset(loadOnlyComponents=set(components_to_load))
    except (ValueError, pyasdm.exceptions.BDFReaderException) as exc:
        raise RuntimeError(
            f"Error in BDFReader.getSubset() for {bdf_reader.getPath()} when loading "
            f"{components_to_load}: {exc}"
        ) from exc

    return subset


def load_subset_with_get_ndarrays(
    bdf_reader: pyasdm.bdf.BDFReader,
    spw_idx: int,
    load_spw_function: Callable,
    load_spw_function_params: tuple,
) -> dict:
    """
    Reads the visibilities of one SPW from the next subset with
    BDFReader.getNDArrays().

    Parameters
    ----------
    bdf_reader : pyasdm.bdf.BDFReader
        Reader of the BDF.
    spw_idx : int
        Index of the SPW in the BDF (counting the SPWs of all basebands).
    load_spw_function : Callable
        Function that loads one SPW from a binary component.
    load_spw_function_params : tuple
        Additional parameters for load_spw_function.

    Returns
    -------
    dict
        The ndarrays returned by BDFReader.getNDArrays() ("visibilities").
    """
    try:
        ndarrays = bdf_reader.getNDArrays(
            arrayNames=["visibilities"],
            spwId=spw_idx,
            loadOneSPWFunction=load_spw_function,
            loadOneSPWFunctionParams=load_spw_function_params,
        )
    except pyasdm.exceptions.BDFReaderException as exc:
        raise RuntimeError(
            f"Error in BDFReader.getNDArrays() for {bdf_reader.getPath()} when "
            f"loading visibilities: {exc}"
        ) from exc

    return ndarrays


def _require_loaded_component(subset: dict, component_name: str) -> np.ndarray:
    component = subset.get(component_name) if isinstance(subset, dict) else None
    if not component or not component["present"]:
        raise RuntimeError(
            f"Binary component '{component_name}' not present in the BDF subset."
        )
    if component["arr"] is None:
        raise RuntimeError(
            f"Binary component '{component_name}' was not loaded from the BDF subset."
        )
    return component["arr"]


def load_vis_subset_from_tree(
    subset: dict,
    guessed_shape: tuple,
    baseband_spw_idxs: tuple[int, int],
    bdf_descr: dict,
    array_slice: tuple[slice, ...],
    out: np.ndarray | None = None,
    layout: BDFSpwLayout | None = None,
) -> np.ndarray:
    """
    Loads the visibilities of one SPW from a subset loaded with
    BDFReader.getSubset(), which has the data of all the SPWs.

    Parameters
    ----------
    subset : dict
        Subset from BDFReader.getSubset(), with the binary components needed by
        the selection loaded.
    guessed_shape : tuple
        Unused (the layout of the data is derived from bdf_descr). Kept for
        compatibility.
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband and index of the SPW within the baseband.
    bdf_descr : dict
        BDF description (from the BDF header).
    array_slice : tuple[slice, ...]
        Selection (time, baseline, frequency, polarization). The time slice is
        relative to the integrations (TIM samples) of the subset.
    out : np.ndarray | None
        complex64 array with the shape of the selection to write into. None: a new
        array.
    layout : BDFSpwLayout | None
        Layout of the SPW in the BDF, when already calculated (calculated from
        bdf_descr if None).

    Returns
    -------
    np.ndarray
        complex64 array (time, baseline, frequency, polarization), with the cross
        baselines followed by the autos: out, if given.
    """
    if layout is None:
        layout = calc_bdf_spw_layout(bdf_descr, baseband_spw_idxs)
    selection = normalize_bdf_selection(
        array_slice, layout, time_len=layout.times_per_subset
    )
    vis = output_array(out, selection_shape(selection), VISIBILITY_DTYPE)

    num_cross_rows = 0
    cross_range = selection.cross_range(layout.num_cross_baselines)
    if cross_range is not None:
        num_cross_rows = _range_len(cross_range)
        load_cross_data_one_spw(
            _require_loaded_component(subset, "crossData"),
            layout,
            selection.time,
            cross_range,
            selection.frequency,
            selection.polarization,
            out=vis[:, :num_cross_rows],
        )

    auto_range = selection.auto_range(layout.num_cross_baselines)
    if auto_range is not None:
        load_auto_data_one_spw(
            _require_loaded_component(subset, "autoData"),
            layout,
            selection.time,
            auto_range,
            selection.frequency,
            selection.polarization,
            out=vis[:, num_cross_rows:],
        )

    return vis


def load_flags_all_subsets_from_trees(
    bdf_reader: pyasdm.bdf.BDFReader,
    guessed_shape: dict[str, tuple[int, ...]],
    bdf_descr: dict,
    baseband_spw_idxs: tuple[int, int],
    array_slice: tuple[slice, ...],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Loads the flags of one SPW from the subsets of a BDF. Needed when the numbers of
    SPWs per baseband or of polarizations per SPW are not uniform, and valid for any
    BDF.

    Parameters
    ----------
    bdf_reader : pyasdm.bdf.BDFReader
        Reader of the BDF, opened and positioned before the first subset.
    guessed_shape : dict[str, tuple[int, ...]]
        Unused (the layout of the flags is derived from bdf_descr). Kept for
        compatibility.
    bdf_descr : dict
        BDF description (from the BDF header).
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband and index of the SPW within the baseband.
    array_slice : tuple[slice, ...]
        Selection (time, baseline, frequency, polarization), as in
        load_visibilities_all_subsets_from_trees. The frequency selection is
        ignored (the BDF flags have no frequency axis).
    out : np.ndarray | None
        bool array (time, baseline, polarization) to write the flags into. None:
        a new array.

    Returns
    -------
    np.ndarray
        bool array (time, baseline, polarization): flagged where the BDF flag word
        is not 0. out, if given.
    """
    layout = calc_bdf_spw_layout(bdf_descr, baseband_spw_idxs)
    selection = normalize_bdf_selection(array_slice, layout)

    def load_subset_block(subset_slice: tuple[slice, ...], block: np.ndarray):
        subset = load_subset_with_get_subset(bdf_reader, ["flags"])
        load_flags_subset_from_tree(
            subset,
            guessed_shape,
            bdf_descr,
            baseband_spw_idxs,
            subset_slice,
            out=block,
            layout=layout,
        )

    return _load_selected_subsets(
        bdf_reader,
        layout,
        selection,
        load_subset_block,
        (_range_len(selection.polarization),),
        FLAG_DTYPE,
        out,
    )


def load_flags_subset_from_tree(
    subset: dict,
    guessed_shape: dict[str, tuple[int, ...]],
    bdf_descr: dict,
    baseband_spw_idxs: tuple[int, int],
    array_slice: tuple[slice, ...],
    out: np.ndarray | None = None,
    layout: BDFSpwLayout | None = None,
) -> np.ndarray:
    """
    Loads the flags of one SPW from one subset of a BDF.

    Parameters
    ----------
    subset : dict
        Subset from BDFReader.getSubset(), with the flags loaded.
    guessed_shape : dict[str, tuple[int, ...]]
        Unused. Kept for compatibility.
    bdf_descr : dict
        BDF description (from the BDF header).
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband and index of the SPW within the baseband.
    array_slice : tuple[slice, ...]
        Selection (time, baseline, frequency, polarization). The time slice is
        relative to the integrations (TIM samples) of the subset. The frequency
        selection is ignored.
    out : np.ndarray | None
        bool array (time, baseline, polarization) of the selected shape to write
        into. None: a new array.
    layout : BDFSpwLayout | None
        Layout of the SPW in the BDF, when already calculated (calculated from
        bdf_descr if None).

    Returns
    -------
    np.ndarray
        bool array (time, baseline, polarization). All False when the subset has
        no flags. out, if given.
    """
    if layout is None:
        layout = calc_bdf_spw_layout(bdf_descr, baseband_spw_idxs)

    flags = subset.get("flags") if isinstance(subset, dict) else None
    if flags and flags["present"]:
        flag_array = _require_loaded_component(subset, "flags")
        baseband_idx, spw_idx = baseband_spw_idxs
        overall_spw_idx = calculate_overall_spw_idx(
            bdf_descr["basebands"], baseband_idx, spw_idx
        )
        offset_additions = calculate_offset_additions_cross_sd(
            bdf_descr,
            baseband_idx,
            overall_spw_idx,
            len(flag_array),
        )

        return load_flags_subset_cross_and_auto_blocks_from_tree(
            flag_array,
            bdf_descr,
            offset_additions,
            guessed_shape,
            baseband_spw_idxs,
            array_slice,
            out=out,
            layout=layout,
        )

    selection = normalize_bdf_selection(
        array_slice, layout, time_len=layout.times_per_subset
    )
    flag_subset = output_array(out, _flag_selection_shape(selection), FLAG_DTYPE)
    flag_subset[...] = False
    return flag_subset


def _flag_selection_shape(selection: BDFSelection) -> tuple[int, int, int]:
    return (
        _range_len(selection.time),
        _range_len(selection.baseline),
        _range_len(selection.polarization),
    )


def load_flags_subset_cross_and_auto_blocks_from_tree(
    flag_array: np.ndarray,
    bdf_descr: dict,
    offset_additions: dict,
    guessed_shape: dict[str, tuple[int, ...]],
    baseband_spw_idxs: tuple[int, int],
    array_slice: tuple[slice, ...],
    out: np.ndarray | None = None,
    layout: BDFSpwLayout | None = None,
) -> np.ndarray:
    """
    Takes the flags of one SPW from the flags binary component of one subset: for
    every integration (TIM sample) a block of flags for the cross baselines (BAL)
    followed by a block for the antennas (ANT).

    Parameters
    ----------
    flag_array : np.ndarray
        flags binary component (int32 flag words).
    bdf_descr : dict
        BDF description (from the BDF header).
    offset_additions : dict
        Offsets "before" and "after" the SPW (or baseband) within the "cross"
        and "auto" rows (see flags_offsets.calculate_offset_additions_cross_sd).
    guessed_shape : dict[str, tuple[int, ...]]
        Unused. Kept for compatibility.
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband and index of the SPW within the baseband.
    array_slice : tuple[slice, ...]
        Selection (time, baseline, frequency, polarization). The time slice is
        relative to the integrations (TIM samples) of the subset. The frequency
        selection is ignored.
    out : np.ndarray | None
        bool array (time, baseline, polarization) of the selected shape to write
        into. None: a new array.
    layout : BDFSpwLayout | None
        Layout of the SPW in the BDF, when already calculated (calculated from
        bdf_descr if None).

    Returns
    -------
    np.ndarray
        bool array (time, baseline, polarization): flagged where the flag word is
        not 0. Full-polarization auto flags (XX, XY, YY) give [XX, XY, XY, YY]
        when the polarization axis has 4 entries. out, if given.
    """
    if layout is None:
        layout = calc_bdf_spw_layout(bdf_descr, baseband_spw_idxs)
    selection = normalize_bdf_selection(
        array_slice, layout, time_len=layout.times_per_subset
    )
    flags = output_array(out, _flag_selection_shape(selection), FLAG_DTYPE)
    num_cross_baselines = layout.num_cross_baselines
    cross_before = (
        int(offset_additions["cross"]["before"]) if num_cross_baselines else 0
    )
    cross_row_len = (
        cross_before + int(offset_additions["cross"]["after"])
        if num_cross_baselines
        else 0
    )
    auto_before = int(offset_additions["auto"]["before"])
    auto_row_len = auto_before + int(offset_additions["auto"]["after"])
    time_stride = (
        num_cross_baselines * cross_row_len + layout.num_antennas * auto_row_len
    )

    time_start, time_stop = selection.time
    pol_start, pol_stop = selection.polarization

    num_cross_rows = 0
    cross_range = selection.cross_range(num_cross_baselines)
    if cross_range is not None:
        row_start, row_stop = cross_range
        num_cross_rows = row_stop - row_start
        words = read_component_block(
            flag_array,
            time_start * time_stride + row_start * cross_row_len + cross_before,
            (time_stop - time_start, num_cross_rows, layout.num_cross_pols),
            (time_stride, cross_row_len, 1),
        )
        np.not_equal(words[..., pol_start:pol_stop], 0, out=flags[:, :num_cross_rows])

    auto_range = selection.auto_range(num_cross_baselines)
    if auto_range is not None:
        row_start, row_stop = auto_range
        words = read_component_block(
            flag_array,
            time_start * time_stride
            + num_cross_baselines * cross_row_len
            + row_start * auto_row_len
            + auto_before,
            (time_stop - time_start, row_stop - row_start, layout.num_sd_pols),
            (time_stride, auto_row_len, 1),
        )
        if layout.num_sd_pols == 3 and layout.num_polarizations == 4:
            # expand XX XY YY => XX XY YX YY (where YX=XY)
            words = words[..., [0, 1, 1, 2]]
        np.not_equal(words[..., pol_start:pol_stop], 0, out=flags[:, num_cross_rows:])

    return flags
