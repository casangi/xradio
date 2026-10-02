"""
Loads the visibilities/flags of one SPW from the 1-D binary component arrays
('arr') returned by pyasdm.bdf.BDFReader.getSubset().

All the data of a subset (for all the SPWs of the BDF) are first read with the
original getSubset() of pyasdm. The 1-D arrays are then reshaped following the
axes order of the BDF specification, and the requested SPW is selected. Reshaping
requires a regular layout: the same number of SPWs in every baseband, and the
same number of channels and polarization products in every SPW. The sizes of the
binary components are checked against the shapes derived from the BDF header,
and an exception is raised when they do not match: data are never silently
truncated or taken from other SPWs.

Layouts of the binary components of one subset (from the BDF specification; the
TIM axis is only present in packed BDFs, i.e. with dimensionality 0):

- crossData: [TIM] BAL BAB SPW [APC] SPP POL, as (real, imaginary) pairs.
- autoData: [TIM] ANT BAB SPW SPP POL, real values. With 3 sdPolProducts
  (XX XY YY) every channel has 4 values: XX, Re(XY), Im(XY), YY.
- flags: for every TIM sample, the block of cross-correlation flags
  (BAL BAB SPW POL) followed by the block of auto-correlation flags
  (ANT BAB SPW POL). Flags can also be given per baseband (no SPW axis) or for
  all the basebands (no BAB and no SPW axes).

The selections follow the conventions of robust_load_data_flags: one key per
dimension (time, baseline, frequency, polarization).

- The time key is local to the BDF and counts integrations (TIM samples): one
  integration per subset for dimensionality 1, numTime integrations per subset
  for packed BDFs (dimensionality 0).
- The baseline axis has the cross baselines in BDF order followed by the
  auto-correlations in antenna order (CROSS_AND_AUTO), or the antennas
  (AUTO_ONLY).
- Only the selected block is materialized, and every dimension is kept: the
  loaders return exactly the selected number of elements along each dimension.
"""

import numpy as np
import pyasdm

from xradio.measurement_set._utils._asdm._utils._bdf.shapes import (
    num_auto_values_per_channel,
    select_apc_index,
    times_per_subset,
)

#: loadOnlyComponents value for BDFReader.getSubset() that skips the reading of
#: every binary component (an empty set would make pyasdm read all of them).
_SKIP_ALL_BINARY_COMPONENTS = frozenset({"__skip_all_binary_components__"})

#: errors raised by pyasdm when a subset cannot be parsed
_SUBSET_READ_ERRORS = (pyasdm.exceptions.BDFReaderException, ValueError, OSError)

_ARRAY_SLICE_ALL = (slice(None), slice(None), slice(None), slice(None))


def load_visibilities_all_subsets(
    bdf_reader: pyasdm.bdf.BDFReader,
    guessed_shape: tuple[int, ...],
    baseband_spw_idxs: tuple[int, int],
    bdf_descr: dict,
    array_slice: tuple[slice | int, ...] | None,
) -> np.ndarray:
    """
    Loads the visibilities of one SPW from a BDF, reshaping the full crossData
    and autoData arrays of the subsets.

    Parameters
    ----------
    bdf_reader : pyasdm.bdf.BDFReader
        Reader of the BDF, opened and positioned before its first subset.
    guessed_shape : tuple[int, ...]
        Shape from define_visibility_shape(): (time, cross_baseline, antenna,
        baseband, spw, frequency, polarization, 2).
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband in the BDF, and index of the SPW in the baseband.
    bdf_descr : dict
        BDF description, see robust_load_data_flags.make_bdf_description().
    array_slice : tuple[slice | int, ...] | None
        Selection (time, baseline, frequency, polarization). Slices must have a
        step of None or 1. The time selection is local to the BDF. An int key
        selects one element and keeps the dimension. None selects everything.

    Returns
    -------
    np.ndarray
        complex64 array (time, baseline, frequency, polarization), with exactly
        the selected lengths. Integer crossData are divided by the scaleFactor of
        the SPW, float crossData are used as is (real-valued crossData of data
        not from the CORRELATOR get an imaginary part 0). With 2 APC values the
        AP_UNCORRECTED data are loaded. Auto-correlations are real, except for 3
        sdPolProducts (XX XY YY), loaded as [XX, XY, conj(XY), YY] when there are
        4 output polarizations (CROSS_AND_AUTO), or as [XX, XY, YY] (AUTO_ONLY).

    Raises
    ------
    NotImplementedError
        If an SPW of the BDF has numBin > 1, or there are several APC values
        without AP_UNCORRECTED.
    ValueError
        If the layout of the BDF is not regular or not supported, a binary
        component is missing or does not have the size expected from the
        header, or the BDF has fewer integrations than selected.
    RuntimeError
        If pyasdm fails to read a subset.
    """
    basebands = bdf_descr["basebands"]
    auto_only = (
        bdf_descr["correlation_mode"] == pyasdm.enumerations.CorrelationMode.AUTO_ONLY
    )
    # a BIN axis in any SPW would change the layout of all the SPWs
    _check_num_bin([spw for bband in basebands for spw in bband["spectralWindows"]])
    _check_regular_layout(basebands, auto_only)

    (
        tims_per_subset,
        cross_baseline_len,
        antenna_len,
        _baseband_len,
        _spw_len,
        frequency_len,
        polarization_len,
        _,
    ) = guessed_shape
    if auto_only:
        cross_baseline_len = 0
    correlator = (
        bdf_descr.get("processor_type", pyasdm.enumerations.ProcessorType.CORRELATOR)
        == pyasdm.enumerations.ProcessorType.CORRELATOR
    )

    spw_descr = _spw_descr(basebands, baseband_spw_idxs)
    sd_polarization_len = len(spw_descr["sdPolProducts"])
    # raises if the auto-correlation products cannot be mapped to the output ones
    _auto_polarization_map(sd_polarization_len, polarization_len)
    apc_idx, apc_len = _find_apc_index(bdf_descr.get("apc"))

    if not array_slice:
        array_slice = _ARRAY_SLICE_ALL
    time_range = _time_range(array_slice[0])
    baseline_slice = _dim_slice(
        array_slice[1], cross_baseline_len + antenna_len, "baseline"
    )
    frequency_slice = _dim_slice(array_slice[2], frequency_len, "frequency")
    polarization_slice = _dim_slice(array_slice[3], polarization_len, "polarization")
    cross_slice, auto_slice = _split_baseline_slice(baseline_slice, cross_baseline_len)

    components = set()
    if cross_slice is not None:
        components.add("crossData")
    if auto_slice is not None:
        components.add("autoData")

    bdf_path = _bdf_path(bdf_reader)
    vis_per_subset = []
    for subset, tim_slice in _iter_selected_subsets(
        bdf_reader, components, time_range, tims_per_subset
    ):
        tim_len = tim_slice.stop - tim_slice.start
        blocks = []
        if cross_slice is not None:
            blocks.append(
                _load_vis_subset_cross_data(
                    _get_component_array(subset, "crossData", bdf_path),
                    guessed_shape,
                    baseband_spw_idxs,
                    (apc_idx, apc_len),
                    spw_descr["scaleFactor"],
                    (tim_slice, cross_slice, frequency_slice, polarization_slice),
                    real_values_allowed=not correlator,
                )
            )
        if auto_slice is not None:
            blocks.append(
                _load_vis_subset_auto_data(
                    _get_component_array(subset, "autoData", bdf_path),
                    guessed_shape,
                    baseband_spw_idxs,
                    sd_polarization_len,
                    (apc_idx, apc_len),
                    (tim_slice, auto_slice, frequency_slice, polarization_slice),
                )
            )
        if blocks:
            vis_subset = np.concatenate(blocks, axis=1)
        else:
            vis_subset = np.zeros(
                (
                    tim_len,
                    0,
                    _slice_len(frequency_slice),
                    _slice_len(polarization_slice),
                ),
                dtype=np.complex64,
            )
        vis_per_subset.append(vis_subset)

    selected_shape = (
        _slice_len(baseline_slice),
        _slice_len(frequency_slice),
        _slice_len(polarization_slice),
    )
    return _concatenate_time(vis_per_subset, selected_shape, np.complex64)


def _load_vis_subset_cross_data(
    cross_data_arr: np.ndarray,
    guessed_shape: tuple[int, ...],
    baseband_spw_idxs: tuple[int, int],
    apc_idx_len: tuple[int, int],
    scale_factor: float | None,
    subset_slice: tuple[slice, slice, slice, slice],
    real_values_allowed: bool = False,
) -> np.ndarray:
    """
    Selects the cross-correlations of one SPW from the crossData array of a
    subset.

    Parameters
    ----------
    cross_data_arr : np.ndarray
        1-D crossData array of the subset (all SPWs), layout
        [TIM] BAL BAB SPW [APC] SPP POL (real, imaginary).
    guessed_shape : tuple[int, ...]
        Shape from define_visibility_shape().
    baseband_spw_idxs : tuple[int, int]
        Baseband index and SPW index in the baseband.
    apc_idx_len : tuple[int, int]
        Index of the APC to load, and number of APC values.
    scale_factor : float | None
        scaleFactor of the SPW, applied to integer data.
    subset_slice : tuple[slice, slice, slice, slice]
        Explicit step-1 slices (TIM within the subset, cross baseline,
        frequency, polarization).
    real_values_allowed : bool
        Accept real-valued crossData (half the size of complex data), as may be
        found in data not from the CORRELATOR. The imaginary part is then 0.

    Returns
    -------
    np.ndarray
        complex64 array (time, cross_baseline, frequency, polarization).
    """
    tims_per_subset, cross_baseline_len, _, baseband_len, spw_len = guessed_shape[:5]
    frequency_len, polarization_len = guessed_shape[5:7]
    apc_idx, apc_len = apc_idx_len
    cross_shape = (
        tims_per_subset,
        cross_baseline_len,
        baseband_len,
        spw_len,
        apc_len,
        frequency_len,
        polarization_len,
        2,
    )
    if real_values_allowed and cross_data_arr.size == _shape_size(cross_shape) // 2:
        cross_shape = cross_shape[:-1] + (1,)
    _check_component_size(cross_data_arr, cross_shape, "crossData")

    time_slice, baseline_slice, frequency_slice, polarization_slice = subset_slice
    cross_values = cross_data_arr.reshape(cross_shape)[
        time_slice,
        baseline_slice,
        baseband_spw_idxs[0],
        baseband_spw_idxs[1],
        apc_idx,
        frequency_slice,
        polarization_slice,
        :,
    ]
    if cross_values.dtype.kind in "iu":
        if not scale_factor:
            raise ValueError(
                f"Integer crossData ({cross_values.dtype}) require a non-zero "
                f"scaleFactor, but the SPW has {scale_factor=}"
            )
        cross_values = cross_values / np.float64(scale_factor)

    vis = np.zeros(cross_values.shape[:-1], dtype=np.complex64)
    vis.real = cross_values[..., 0]
    if cross_values.shape[-1] == 2:
        vis.imag = cross_values[..., 1]
    return vis


def _load_vis_subset_auto_data(
    auto_data_arr: np.ndarray,
    guessed_shape: tuple[int, ...],
    baseband_spw_idxs: tuple[int, int],
    sd_polarization_len: int,
    apc_idx_len: tuple[int, int],
    subset_slice: tuple[slice, slice, slice, slice],
) -> np.ndarray:
    """
    Selects the auto-correlations of one SPW from the autoData array of a
    subset.

    Parameters
    ----------
    auto_data_arr : np.ndarray
        1-D autoData array of the subset (all SPWs), layout
        [TIM] ANT BAB SPW SPP POL. An APC axis (between SPW and SPP) is also
        accepted when there are several APC values.
    guessed_shape : tuple[int, ...]
        Shape from define_visibility_shape(). Its polarization length is the
        number of output polarizations.
    baseband_spw_idxs : tuple[int, int]
        Baseband index and SPW index in the baseband.
    sd_polarization_len : int
        Number of sdPolProducts of the SPW.
    apc_idx_len : tuple[int, int]
        Index of the APC to load, and number of APC values.
    subset_slice : tuple[slice, slice, slice, slice]
        Explicit step-1 slices (TIM within the subset, antenna, frequency,
        output polarization).

    Returns
    -------
    np.ndarray
        complex64 array (time, antenna, frequency, polarization).
    """
    tims_per_subset, _, antenna_len, baseband_len, spw_len = guessed_shape[:5]
    frequency_len, polarization_len = guessed_shape[5:7]
    values_per_channel = num_auto_values_per_channel(sd_polarization_len)
    auto_shape = (
        tims_per_subset,
        antenna_len,
        baseband_len,
        spw_len,
        1,
        frequency_len,
        values_per_channel,
    )
    apc_idx, apc_len = apc_idx_len
    if apc_len > 1 and auto_data_arr.size == _shape_size(auto_shape) * apc_len:
        auto_shape = auto_shape[:4] + (apc_len,) + auto_shape[5:]
    else:
        apc_idx = 0
    _check_component_size(auto_data_arr, auto_shape, "autoData")

    time_slice, antenna_slice, frequency_slice, polarization_slice = subset_slice
    auto_values = auto_data_arr.reshape(auto_shape)[
        time_slice,
        antenna_slice,
        baseband_spw_idxs[0],
        baseband_spw_idxs[1],
        apc_idx,
        frequency_slice,
        :,
    ]

    if sd_polarization_len == 3:
        # autoData: "parallel-hand polarizations are real-valued, while
        # cross-hand polarizations are complex-valued" (BDF specification).
        # For an auto-correlation YX = conj(XY).
        xy_values = auto_values[..., 1] + 1j * auto_values[..., 2]
        if polarization_len == 4:
            products = [
                auto_values[..., 0],
                xy_values,
                np.conj(xy_values),
                auto_values[..., 3],
            ]
        else:
            products = [auto_values[..., 0], xy_values, auto_values[..., 3]]
        selected_products = products[polarization_slice]
        if not selected_products:
            return np.zeros(auto_values.shape[:-1] + (0,), dtype=np.complex64)
        vis = np.stack(selected_products, axis=-1)
    else:
        vis = auto_values[..., polarization_slice]

    return vis.astype(np.complex64)


def define_visibility_shape(
    bdf_descr: dict, baseband_spw_idxs: tuple[int, int]
) -> tuple[int, ...]:
    """
    Defines the shape used to load the crossData/autoData binary components of
    the subsets of a BDF, for one SPW.

    Parameters
    ----------
    bdf_descr : dict
        BDF description, see robust_load_data_flags.make_bdf_description().
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband in the BDF, and index of the SPW in the baseband.

    Returns
    -------
    tuple[int, ...]
        (time, cross_baseline, antenna, baseband, spw, frequency, polarization,
        2), where time is the number of integrations (TIM samples) per subset
        (numTime for packed BDFs, 1 otherwise), spw is the number of SPWs in the
        baseband, frequency is the numSpectralPoint of the SPW, and polarization
        is the number of output polarization products (crossPolProducts, or
        sdPolProducts for AUTO_ONLY data).

    Raises
    ------
    NotImplementedError
        If the SPW has numBin > 1 (the BIN axis is not supported).
    """
    basebands = bdf_descr["basebands"]
    _check_num_bin([_spw_descr(basebands, baseband_spw_idxs)])

    baseband_len = len(basebands)
    antenna_len = bdf_descr["num_antenna"]
    cross_baseline_len = antenna_len * (antenna_len - 1) // 2
    baseband_descr = basebands[baseband_spw_idxs[0]]
    spw_len = len(baseband_descr["spectralWindows"])
    spw_descr = baseband_descr["spectralWindows"][baseband_spw_idxs[1]]
    frequency_len = spw_descr["numSpectralPoint"]
    if bdf_descr["correlation_mode"] == pyasdm.enumerations.CorrelationMode.AUTO_ONLY:
        polarization_len = len(spw_descr["sdPolProducts"])
    else:
        polarization_len = len(spw_descr["crossPolProducts"]) or len(
            spw_descr["sdPolProducts"]
        )

    shape = (
        times_per_subset(bdf_descr),
        cross_baseline_len,
        antenna_len,
        baseband_len,
        spw_len,
        frequency_len,
        polarization_len,
        2,
    )

    return shape


def load_flags_all_subsets(
    bdf_reader: pyasdm.bdf.BDFReader,
    guessed_shape: dict[str, tuple[int, ...]],
    baseband_spw_idxs: tuple[int, int],
    array_slice: tuple[slice | int, ...] | None,
) -> np.ndarray:
    """
    Loads the flags of one SPW from a BDF, reshaping the full flags arrays of
    the subsets.

    Parameters
    ----------
    bdf_reader : pyasdm.bdf.BDFReader
        Reader of the BDF, opened and positioned before its first subset.
    guessed_shape : dict[str, tuple[int, ...]]
        Shapes from define_flag_shape().
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband in the BDF, and index of the SPW in the baseband.
    array_slice : tuple[slice | int, ...] | None
        Selection (time, baseline, frequency, polarization). Slices must have a
        step of None or 1. The time selection is local to the BDF. An int key
        selects one element and keeps the dimension. The frequency key is not
        used (the BDF flags have no frequency axis). None selects everything.

    Returns
    -------
    np.ndarray
        bool array (time, baseline, polarization), with exactly the selected
        lengths. A sample is flagged when its BDF flag word is not 0. For 3
        auto-correlation products (XX XY YY) with 4 cross products, the XY
        flag is also used for YX. Subsets without flags are not flagged.

    Raises
    ------
    ValueError
        If the size of a flags array does not match any supported layout, the
        polarization products are not supported, or the BDF has fewer
        integrations than selected.
    RuntimeError
        If pyasdm fails to read a subset.
    """
    shape_cross = guessed_shape["cross"]
    shape_auto = guessed_shape["auto"]
    tims_per_subset, antenna_len = shape_auto[:2]
    cross_baseline_len = shape_cross[1] if shape_cross else 0
    polarization_len = shape_cross[-1] if shape_cross else shape_auto[-1]
    _auto_polarization_map(shape_auto[-1], polarization_len)

    if not array_slice:
        array_slice = _ARRAY_SLICE_ALL
    time_range = _time_range(array_slice[0])
    baseline_slice = _dim_slice(
        array_slice[1], cross_baseline_len + antenna_len, "baseline"
    )
    polarization_slice = _dim_slice(array_slice[3], polarization_len, "polarization")

    flag_per_subset = [
        _load_flags_subset(
            subset,
            guessed_shape,
            baseband_spw_idxs,
            (tim_slice, baseline_slice, polarization_slice),
        )
        for subset, tim_slice in _iter_selected_subsets(
            bdf_reader, {"flags"}, time_range, tims_per_subset
        )
    ]

    selected_shape = (_slice_len(baseline_slice), _slice_len(polarization_slice))
    return _concatenate_time(flag_per_subset, selected_shape, bool)


def define_flag_shape(
    bdf_descr: dict, baseband_spw_idxs: tuple[int, int]
) -> dict[str, tuple[int, ...]]:
    """
    Defines the shapes of the cross-correlation and auto-correlation blocks of
    the flags binary component of the subsets of a BDF (one flag per SPW).

    Parameters
    ----------
    bdf_descr : dict
        BDF description, see robust_load_data_flags.make_bdf_description().
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband in the BDF, and index of the SPW in the baseband.

    Returns
    -------
    dict[str, tuple[int, ...]]
        "cross": (time, cross_baseline, baseband, spw, crossPolProducts), or ()
        for AUTO_ONLY data; "auto": (time, antenna, baseband, spw,
        sdPolProducts). time is the number of integrations (TIM samples) per
        subset and spw the number of SPWs in the baseband of the selected SPW.
    """
    baseband_len = len(bdf_descr["basebands"])
    antenna_len = bdf_descr["num_antenna"]
    baseline_len = antenna_len * (antenna_len - 1) // 2
    baseband_descr = bdf_descr["basebands"][baseband_spw_idxs[0]]
    spw_len = len(baseband_descr["spectralWindows"])
    spw_descr = baseband_descr["spectralWindows"][baseband_spw_idxs[1]]
    cross_pol_len = len(spw_descr["crossPolProducts"])
    auto_pol_len = len(spw_descr["sdPolProducts"])
    time_len = times_per_subset(bdf_descr)

    if bdf_descr["correlation_mode"] == pyasdm.enumerations.CorrelationMode.AUTO_ONLY:
        shape_cross = ()
    else:
        shape_cross = (
            time_len,
            baseline_len,
            baseband_len,
            spw_len,
            cross_pol_len,
        )
    shape_auto = (time_len, antenna_len, baseband_len, spw_len, auto_pol_len)

    return {
        "cross": shape_cross,
        "auto": shape_auto,
    }


def _find_flags_layout(
    guessed_shape: dict[str, tuple[int, ...]],
    flags_size: int,
    baseband_spw_idxs: tuple[int, int],
) -> tuple[dict[str, tuple[int, ...]], tuple[int, int]]:
    """
    Finds the layout of the flags binary component of a subset from its size.

    The supported layouts are tried in this order: one flag per SPW (as given by
    define_flag_shape()), one flag per baseband (no SPW axis), and one flag for
    all the basebands (no BAB and no SPW axes).

    Parameters
    ----------
    guessed_shape : dict[str, tuple[int, ...]]
        Shapes from define_flag_shape().
    flags_size : int
        Size of the flags array of the subset.
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband in the BDF, and index of the SPW in the baseband.

    Returns
    -------
    tuple[dict[str, tuple[int, ...]], tuple[int, int]]
        Shapes of the cross and auto blocks for the layout found (same
        structure as guessed_shape, the baseband/spw axes having length 1 when
        absent from the layout), and the (baseband, spw) indices to use with
        these shapes.

    Raises
    ------
    ValueError
        If flags_size does not match any of the supported layouts.
    """
    baseband_idx, spw_idx = baseband_spw_idxs
    candidates = [
        ("one flag per SPW", True, True, (baseband_idx, spw_idx)),
        ("one flag per baseband", True, False, (baseband_idx, 0)),
        ("one flag for all basebands", False, False, (0, 0)),
    ]
    tried = []
    for description, keep_baseband, keep_spw, idxs in candidates:
        shapes = {
            name: _reduce_flag_shape(shape, keep_baseband, keep_spw)
            for name, shape in guessed_shape.items()
        }
        size = _shape_size(shapes["cross"]) + _shape_size(shapes["auto"])
        if size == flags_size:
            return shapes, idxs
        tried.append(f"{description}: {size}")

    raise ValueError(
        f"Unexpected size of the flags binary component: {flags_size}. Sizes "
        f"expected from the BDF header ({guessed_shape=}) are: " + ", ".join(tried)
    )


def _load_flags_subset(
    subset: dict,
    guessed_shape: dict[str, tuple[int, ...]],
    baseband_spw_idxs: tuple[int, int],
    subset_slice: tuple[slice, slice, slice],
) -> np.ndarray:
    """
    Loads the flags of one SPW from one subset of a BDF.

    The flags array of the subset (all SPWs) is reshaped as per the layout found
    from its size, the SPW is selected, and the flag words are converted to
    bool (flagged when the word is not 0).

    Parameters
    ----------
    subset : dict
        Subset, as returned by BDFReader.getSubset().
    guessed_shape : dict[str, tuple[int, ...]]
        Shapes from define_flag_shape().
    baseband_spw_idxs : tuple[int, int]
        Index of the baseband in the BDF, and index of the SPW in the baseband.
    subset_slice : tuple[slice, slice, slice]
        Explicit step-1 slices (TIM within the subset, baseline, polarization).

    Returns
    -------
    np.ndarray
        bool array (time, baseline, polarization). The frequency dimension is
        not included, as the BDF flags are not given per channel. All False if
        the subset has no flags.
    """
    time_slice, baseline_slice, polarization_slice = subset_slice
    shape_cross = guessed_shape["cross"]
    shape_auto = guessed_shape["auto"]
    cross_baseline_len = shape_cross[1] if shape_cross else 0
    polarization_len = shape_cross[-1] if shape_cross else shape_auto[-1]
    out_shape = (
        _slice_len(time_slice),
        _slice_len(baseline_slice),
        _slice_len(polarization_slice),
    )

    flags = subset.get("flags")
    if not flags or not flags.get("present") or flags.get("arr") is None:
        return np.zeros(out_shape, dtype=bool)

    flag_words = np.asarray(flags["arr"])
    shapes, (baseband_idx, spw_idx) = _find_flags_layout(
        guessed_shape, flag_words.size, baseband_spw_idxs
    )
    tims_per_subset = shapes["auto"][0]
    flag_words = flag_words.reshape(tims_per_subset, -1)[time_slice]
    cross_len = _shape_size(shapes["cross"][1:]) if shapes["cross"] else 0

    cross_slice, auto_slice = _split_baseline_slice(baseline_slice, cross_baseline_len)
    blocks = []
    if cross_slice is not None:
        cross_words = flag_words[:, :cross_len].reshape(
            (out_shape[0], *shapes["cross"][1:])
        )
        blocks.append(
            cross_words[:, cross_slice, baseband_idx, spw_idx, polarization_slice] != 0
        )
    if auto_slice is not None:
        auto_words = flag_words[:, cross_len:].reshape(
            (out_shape[0], *shapes["auto"][1:])
        )
        polarization_map = _auto_polarization_map(shapes["auto"][-1], polarization_len)[
            polarization_slice
        ]
        blocks.append(
            auto_words[:, auto_slice, baseband_idx, spw_idx][..., polarization_map] != 0
        )

    if not blocks:
        return np.zeros(out_shape, dtype=bool)
    return np.concatenate(blocks, axis=1)


def _auto_polarization_map(
    sd_polarization_len: int, polarization_len: int
) -> np.ndarray:
    """
    Indices of the auto-correlation products (sdPolProducts) used for every
    output polarization: identity, or (XX, XY, XY, YY) for 3 sd products (XX
    XY YY) and 4 output (cross-correlation) products.

    Raises
    ------
    ValueError
        For any other combination of numbers of products.
    """
    if sd_polarization_len == polarization_len:
        return np.arange(polarization_len)
    if sd_polarization_len == 3 and polarization_len == 4:
        return np.array([0, 1, 1, 2])
    raise ValueError(
        f"Unsupported polarization products: {sd_polarization_len} "
        f"auto-correlation products with {polarization_len} cross-correlation "
        "products"
    )


def _check_num_bin(spws: list[dict]):
    """
    Raises NotImplementedError if any of the SPW descriptions has more than one
    bin (BIN axis, as in switching modes).
    """
    num_bins = [spw.get("numBin") or 1 for spw in spws]
    if any(num_bin > 1 for num_bin in num_bins):
        raise NotImplementedError(
            "Loading BDF data with numBin > 1 (BIN axis, e.g. switching modes) is "
            f"not supported. numBin per SPW: {num_bins}"
        )


def _check_regular_layout(basebands: list[dict], auto_only: bool):
    """
    Raises ValueError if the SPWs of a BDF do not have a regular layout (same
    number of SPWs in every baseband, same numbers of channels and polarization
    products in every SPW), as required to reshape the binary components. The
    crossPolProducts are not used for AUTO_ONLY data.
    """
    spws = [spw for bband in basebands for spw in bband["spectralWindows"]]
    layout_values = {
        "SPWs per baseband": {len(bband["spectralWindows"]) for bband in basebands},
        "channels": {spw["numSpectralPoint"] for spw in spws},
        "sdPolProducts": {len(spw["sdPolProducts"]) for spw in spws},
    }
    if not auto_only:
        layout_values["crossPolProducts"] = {
            len(spw["crossPolProducts"]) for spw in spws
        }
    irregular = {
        name: values for name, values in layout_values.items() if len(values) > 1
    }
    if irregular:
        raise ValueError(
            "Loading by reshaping the binary components requires the same number "
            "of SPWs in every baseband and the same numbers of channels and "
            f"polarization products in every SPW. Found different values: {irregular}"
        )


def _find_apc_index(apc_list: list | None) -> tuple[int, int]:
    """
    Index of the APC to load (AP_UNCORRECTED when there are several APC values,
    as the CASA importasdm default) and number of APC values.

    Raises
    ------
    NotImplementedError
        If there are several APC values and none is AP_UNCORRECTED.
    """
    apc_list = list(apc_list or [])
    return select_apc_index(apc_list, warn=False), max(len(apc_list), 1)


def _spw_descr(basebands: list[dict], baseband_spw_idxs: tuple[int, int]) -> dict:
    """Description of an SPW from the basebands list of a BDF header."""
    return basebands[baseband_spw_idxs[0]]["spectralWindows"][baseband_spw_idxs[1]]


def _reduce_flag_shape(
    shape: tuple[int, ...], keep_baseband: bool, keep_spw: bool
) -> tuple[int, ...]:
    """Flag block shape (time, bl|ant, bb, spw, pol) with bb/spw axes set to 1."""
    if not shape:
        return ()
    baseband_len = shape[2] if keep_baseband else 1
    spw_len = shape[3] if keep_spw else 1
    return (shape[0], shape[1], baseband_len, spw_len, shape[4])


def _shape_size(shape: tuple[int, ...]) -> int:
    """Number of elements of a shape, 0 for an empty (absent block) shape."""
    return int(np.prod(shape, dtype=np.int64)) if shape else 0


def _check_component_size(arr: np.ndarray, shape: tuple[int, ...], name: str):
    """Raises ValueError if a binary component array does not have the size of shape."""
    if arr.size != _shape_size(shape):
        raise ValueError(
            f"Unexpected size of the {name} binary component: {arr.size}, expected "
            f"{_shape_size(shape)} for shape {shape} (derived from the BDF header)"
        )


def _dim_slice(key: slice | int | None, dim_len: int, dim_name: str) -> slice:
    """
    Normalizes the key of one dimension of known length to an explicit slice
    slice(start, stop) with 0 <= start <= stop <= dim_len.

    Raises
    ------
    IndexError
        If an int key is out of range.
    ValueError
        If a slice has a step other than None or 1.
    TypeError
        If the key is neither None, int nor slice.
    """
    if key is None:
        return slice(0, dim_len)
    if isinstance(key, int | np.integer):
        idx = int(key)
        if idx < 0:
            idx += dim_len
        if not 0 <= idx < dim_len:
            raise IndexError(
                f"Index {key} out of range for {dim_name} of length {dim_len}"
            )
        return slice(idx, idx + 1)
    if isinstance(key, slice):
        if key.step not in (None, 1):
            raise ValueError(
                f"Unsupported {dim_name} selection {key}: only steps of 1 are supported"
            )
        start, stop, _ = key.indices(dim_len)
        return slice(start, max(start, stop))
    raise TypeError(f"Unsupported {dim_name} selection {key!r} ({type(key)})")


def _time_range(key: slice | int | None) -> tuple[int, int | None]:
    """
    Normalizes the BDF-local time key to (start, stop), stop being None when the
    selection extends to the end of the BDF (its length is not known before
    reading it).
    """
    if key is None:
        return 0, None
    if isinstance(key, int | np.integer):
        if key < 0:
            raise ValueError(f"Negative BDF-local time index not supported: {key}")
        return int(key), int(key) + 1
    if isinstance(key, slice):
        if key.step not in (None, 1):
            raise ValueError(
                f"Unsupported time selection {key}: only steps of 1 are supported"
            )
        start = 0 if key.start is None else int(key.start)
        stop = None if key.stop is None else int(key.stop)
        if start < 0 or (stop is not None and stop < 0):
            raise ValueError(f"Negative BDF-local time indices not supported: {key}")
        if stop is not None:
            stop = max(start, stop)
        return start, stop
    raise TypeError(f"Unsupported time selection {key!r} ({type(key)})")


def _slice_len(dim_slice: slice) -> int:
    """Length of an explicit step-1 slice."""
    return dim_slice.stop - dim_slice.start


def _split_baseline_slice(
    baseline_slice: slice, cross_baseline_len: int
) -> tuple[slice | None, slice | None]:
    """
    Splits an explicit slice of the baseline axis (cross baselines followed by
    the auto-correlations) into the slices of the cross baselines and of the
    antennas. None when no element of that block is selected.
    """
    start, stop = baseline_slice.start, baseline_slice.stop
    cross_slice = None
    auto_slice = None
    if start < min(stop, cross_baseline_len):
        cross_slice = slice(start, min(stop, cross_baseline_len))
    if stop > max(start, cross_baseline_len):
        auto_slice = slice(
            max(start, cross_baseline_len) - cross_baseline_len,
            stop - cross_baseline_len,
        )
    return cross_slice, auto_slice


def _bdf_path(bdf_reader: pyasdm.bdf.BDFReader) -> str:
    """Path of the BDF, for messages."""
    try:
        return str(bdf_reader.getPath())
    except AttributeError:
        return "<unknown BDF path>"


def _get_subset(
    bdf_reader: pyasdm.bdf.BDFReader, components: set[str], subset_idx: int
) -> dict:
    """
    Reads the next subset of a BDF, loading only the given binary components.

    Raises
    ------
    RuntimeError
        If pyasdm fails to read the subset (the message names the BDF).
    """
    try:
        return bdf_reader.getSubset(loadOnlyComponents=components)
    except _SUBSET_READ_ERRORS as exc:
        raise RuntimeError(
            f"Error reading subset #{subset_idx} of BDF {_bdf_path(bdf_reader)} with "
            f"BDFReader.getSubset(): {exc!r}"
        ) from exc


def _iter_selected_subsets(
    bdf_reader: pyasdm.bdf.BDFReader,
    components: set[str],
    time_range: tuple[int, int | None],
    tims_per_subset: int,
):
    """
    Iterates over the subsets of a BDF that hold the selected integrations.

    Subsets before the selection are skipped without reading their binary
    components, and the reading stops after the last selected integration.

    Parameters
    ----------
    bdf_reader : pyasdm.bdf.BDFReader
        Reader of the BDF, positioned before its first subset.
    components : set[str]
        Binary components to load. If empty, no component is read (only the
        number of integrations is used).
    time_range : tuple[int, int | None]
        BDF-local (start, stop) integration indices, stop None for "until the
        end".
    tims_per_subset : int
        Number of integrations (TIM samples) per subset.

    Yields
    ------
    tuple[dict, slice]
        The subset and the explicit slice of its TIM samples selected.

    Raises
    ------
    ValueError
        If the BDF ends before the stop of the time range.
    """
    time_start, time_stop = time_range
    if tims_per_subset < 1:
        raise ValueError(
            f"Invalid number of integrations per subset ({tims_per_subset}) in BDF "
            f"{_bdf_path(bdf_reader)}"
        )
    components = set(components) or set(_SKIP_ALL_BINARY_COMPONENTS)

    first_tim = 0
    subset_idx = 0
    while (time_stop is None or first_tim < time_stop) and bdf_reader.hasSubset():
        end_tim = first_tim + tims_per_subset
        if end_tim <= time_start:
            _get_subset(bdf_reader, _SKIP_ALL_BINARY_COMPONENTS, subset_idx)
        else:
            subset = _get_subset(bdf_reader, components, subset_idx)
            local_stop = min(end_tim if time_stop is None else time_stop, end_tim)
            yield (
                subset,
                slice(max(time_start, first_tim) - first_tim, local_stop - first_tim),
            )
        first_tim = end_tim
        subset_idx += 1

    if time_stop is not None and first_tim < time_stop:
        raise ValueError(
            f"BDF {_bdf_path(bdf_reader)} has {first_tim} integrations, but "
            f"integrations [{time_start}, {time_stop}) were selected"
        )


def _get_component_array(subset: dict, name: str, bdf_path: str) -> np.ndarray:
    """
    The 1-D array of a binary component of a subset.

    Raises
    ------
    ValueError
        If the component is not present or was not loaded.
    """
    component = subset.get(name)
    if not component or not component.get("present") or component.get("arr") is None:
        # autoData is always present in ALMA BDFs (BDF specification)
        raise ValueError(
            f"Binary component {name!r} not present in a subset of BDF {bdf_path}"
        )
    return np.asarray(component["arr"])


def _concatenate_time(
    per_subset: list[np.ndarray], selected_shape: tuple[int, ...], dtype
) -> np.ndarray:
    """Concatenates the per-subset arrays along time (empty array if none)."""
    if not per_subset:
        return np.zeros((0, *selected_shape), dtype=dtype)
    return np.concatenate(per_subset, axis=0)
