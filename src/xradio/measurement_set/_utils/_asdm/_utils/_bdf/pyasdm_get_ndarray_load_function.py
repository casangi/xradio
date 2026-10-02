"""
Module that could be moved to a BDFReader.getNDArray() method in pyasdm or could
stay here, as a function injected into BDFReader.getNDArrays() (the
"loadOneSPWFunction").

Defines the function that loads the visibilities of one SPW from the crossData and
autoData binary components, used by BDFReader.getNDArrays(), and the decoders it
uses. The decoders read the data of one SPW either directly from the BDF file
(getNDArrays) or from the 1D arrays loaded by BDFReader.getSubset(). They read
contiguous blocks of rows (baselines or antennas, of bounded size) instead of one
read per baseline/antenna, select the SPW / channels / polarizations with array
views, and produce complex64 visibilities. All positions are computed in values
of the binary component and converted to bytes (itemsize) for file reads.

The decoders can write into a preallocated output array (``out``), so that the
cross and auto rows of a subset go straight into their place in the result of a
load, without per-component arrays and without the concatenation done by
BDFReader.getNDArrays() when the loader function returns arrays.
"""

import os
import typing
from collections.abc import Callable

import numpy as np

from xradio.measurement_set._utils._asdm._utils._bdf.basebands_spws import (
    find_spw_in_basebands_list,
)
from xradio.measurement_set._utils._asdm._utils._bdf.shapes import (
    BDFSelection,
    BDFSpwLayout,
    calc_bdf_spw_layout,
    normalize_bdf_selection,
)

#: dtype of the visibilities produced by the loaders
VISIBILITY_DTYPE = np.dtype(np.complex64)
#: Maximum number of values read from a binary component at once (memory bound of
#: the reads, which span the data of all the SPWs of the rows read)
MAX_VALUES_PER_READ = 4 * 1024**2


def output_array(
    out: np.ndarray | None, shape: tuple[int, ...], dtype: np.dtype
) -> np.ndarray:
    """
    Returns the array the loaders write into: out (checked), or a new (empty) array.

    Parameters
    ----------
    out : np.ndarray | None
        Preallocated output array (can be a view of a larger array), or None.
    shape : tuple[int, ...]
        Expected shape.
    dtype : np.dtype
        Expected dtype.

    Returns
    -------
    np.ndarray
        out, or a new uninitialized array of the given shape and dtype.

    Raises
    ------
    ValueError
        If out does not have the expected shape and dtype or is not writeable.
    """
    if out is None:
        return np.empty(shape, dtype=dtype)
    if (
        not isinstance(out, np.ndarray)
        or out.shape != tuple(shape)
        or out.dtype != np.dtype(dtype)
        or not out.flags.writeable
    ):
        raise ValueError(
            f"Output array with shape {getattr(out, 'shape', None)} and dtype "
            f"{getattr(out, 'dtype', None)}, expected a writeable array with shape "
            f"{tuple(shape)} and dtype {np.dtype(dtype)}"
        )
    return out


def selection_shape(selection: BDFSelection) -> tuple[int, int, int, int]:
    """
    Shape (time, baseline, frequency, polarization) of a selection with explicit
    ranges (time stop not None).
    """
    return (
        _range_len(selection.time),
        _range_len(selection.baseline),
        _range_len(selection.frequency),
        _range_len(selection.polarization),
    )


def load_visibilities_one_spw_to_ndarray(
    component_name: str,
    overall_spw_idx: int,
    bdf_file: typing.BinaryIO,
    data_type: np.dtype,
    elements_count: int,
    bdf_descr: dict,
    components_to_load: list[str],
    guessed_shape: tuple[int, ...],
    array_slice: tuple[slice, ...],
    out: np.ndarray | None = None,
    layout: BDFSpwLayout | None = None,
    loaded_components: set[str] | None = None,
) -> np.ndarray | None:
    """
    Function meant to be passed to pyasdm.BDFReader.getNDArrays as loader function
    that does "load data from one SPW only, skipping all other SPWs".

    pyasdm.BDFReader.getNDArrays calls it for the crossData and autoData binary
    components of one subset, with the file positioned at the beginning of the
    component. Without ``out`` it returns the selected rows of the component, and
    getNDArrays concatenates the results along the baseline dimension (cross
    baselines first, then autos) into an MSv4 style ndarray (time, baseline,
    frequency, polarization). With ``out`` (preferred, as it avoids the
    per-component arrays and the concatenated copy) the rows are written into
    their place in ``out`` and None is returned, so getNDArrays gives None as
    "visibilities" and the data are in ``out``.

    The parameters after elements_count are given to getNDArrays as
    loadOneSPWFunctionParams (in this order).

    Parameters
    ----------
    component_name : str
        Binary component: "crossData" or "autoData".
    overall_spw_idx : int
        Index of the SPW in the BDF (counting the SPWs of all basebands).
    bdf_file : typing.BinaryIO
        BDF file, positioned at the beginning of the binary component.
    data_type : np.dtype
        dtype of the values of the binary component (with the BDF byte order).
    elements_count : int
        Number of values in the binary component (from the BDF header).
    bdf_descr : dict
        BDF description (from the BDF header).
    components_to_load : list[str]
        Binary components to load. None is returned for other components.
    guessed_shape : tuple[int, ...]
        Unused (the layout of the data is derived from bdf_descr). Kept for
        compatibility.
    array_slice : tuple[slice, ...]
        Selection (time, baseline, frequency, polarization). The time slice is
        relative to the integrations (TIM samples) of the subset. The baseline
        slice applies to the cross baselines followed by the autos.
    out : np.ndarray | None
        complex64 array (time, baseline, frequency, polarization) with the shape of
        the whole selection (cross and auto rows), where the selected rows of the
        component are written. None to return them in a new array.
    layout : BDFSpwLayout | None
        Layout of the SPW overall_spw_idx in the BDF, when already calculated
        (calculated from bdf_descr if None).
    loaded_components : set[str] | None
        If given, the name of the component is added to it when data are loaded
        from the component (to check that every component needed was present in
        the subset).

    Returns
    -------
    np.ndarray | None
        complex64 array (time, baseline, frequency, polarization) with the
        selected data of the component. None if nothing is selected from it, or
        if the data were written into ``out``.
    """
    if component_name not in components_to_load:
        return None

    if layout is None:
        baseband_spw_idxs = find_spw_in_basebands_list(
            overall_spw_idx, bdf_descr["basebands"], getattr(bdf_file, "name", "")
        )
        layout = calc_bdf_spw_layout(bdf_descr, baseband_spw_idxs)
    selection = normalize_bdf_selection(
        array_slice, layout, time_len=layout.times_per_subset
    )

    cross_range = selection.cross_range(layout.num_cross_baselines)
    if component_name == "crossData":
        row_range = cross_range
        first_out_row = 0
        load_function = load_cross_data_one_spw
    elif component_name == "autoData":
        row_range = selection.auto_range(layout.num_cross_baselines)
        first_out_row = 0 if cross_range is None else _range_len(cross_range)
        load_function = load_auto_data_one_spw
    else:
        raise ValueError(f"Unexpected binary component: {component_name}")

    if row_range is None:
        return None

    component_out = None
    if out is not None:
        out = output_array(out, selection_shape(selection), VISIBILITY_DTYPE)
        component_out = out[:, first_out_row : first_out_row + _range_len(row_range)]

    vis = load_function(
        bdf_file,
        layout,
        selection.time,
        row_range,
        selection.frequency,
        selection.polarization,
        data_type=np.dtype(data_type),
        elements_count=elements_count,
        out=component_out,
    )
    if loaded_components is not None:
        loaded_components.add(component_name)

    return None if out is not None else vis


def read_component_block(
    source: np.ndarray | typing.BinaryIO,
    start: int,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    data_type: np.dtype | None = None,
    elements_count: int | None = None,
    component_offset: int | None = None,
) -> np.ndarray:
    """
    Reads a block of values of a binary component with one contiguous read and
    returns it as a (read-only) strided view.

    Element ``idx`` of the returned array is the value at position ``start +
    sum(idx * strides)`` of the binary component.

    Parameters
    ----------
    source : np.ndarray | typing.BinaryIO
        Values of the binary component (1D array), or BDF file.
    start : int
        Position (in values) of the first value of the block.
    shape : tuple[int, ...]
        Shape of the block.
    strides : tuple[int, ...]
        Strides of the block, in values.
    data_type : np.dtype | None
        dtype of the values. Required when reading from a file.
    elements_count : int | None
        Number of values of the binary component. Required when reading from a
        file.
    component_offset : int | None
        Position (bytes) of the binary component in the file. None: the current
        position of the file.

    Returns
    -------
    np.ndarray
        Read-only array of the given shape.
    """
    if isinstance(source, np.ndarray):
        elements_count = source.size
    stop = (
        start
        + sum((num - 1) * stride for num, stride in zip(shape, strides, strict=True))
        + 1
    )
    if start < 0 or stop > elements_count:
        raise ValueError(
            f"Block [{start}, {stop}) out of the binary component, which has "
            f"{elements_count} values"
        )

    if isinstance(source, np.ndarray):
        values = source[start:stop]
    else:
        if component_offset is None:
            component_offset = source.tell()
        source.seek(component_offset + start * data_type.itemsize, os.SEEK_SET)
        values = np.fromfile(source, dtype=data_type, count=stop - start)
        if values.size != stop - start:
            raise ValueError(
                f"Unexpected end of BDF file: read {values.size} values instead of "
                f"{stop - start}"
            )

    return np.lib.stride_tricks.as_strided(
        values,
        shape=shape,
        strides=tuple(stride * values.strides[0] for stride in strides),
        writeable=False,
    )


def _check_component_size(
    component_name: str, elements_count: int, expected_count: int
):
    if elements_count != expected_count:
        raise ValueError(
            f"The {component_name} binary component has {elements_count} values but "
            f"{expected_count} are expected from the BDF header description "
            "(numAntenna, SPWs, numSpectralPoint, polarization products, APC list, "
            "numBin, numTime). Unsupported or inconsistent BDF data layout."
        )


def _process_row_blocks(
    source: np.ndarray | typing.BinaryIO,
    num_rows: int,
    row_len: int,
    spw_offset: int,
    values_per_channel: int,
    time_range: tuple[int, int],
    row_range: tuple[int, int],
    frequency_range: tuple[int, int],
    data_type: np.dtype | None,
    elements_count: int | None,
    process_block: Callable[[int, slice, np.ndarray], None],
):
    """
    Reads the values of one SPW for a range of TIM samples and rows (baselines or
    antennas) and channels, in contiguous blocks of rows. Every read spans whole
    rows (the data of all the SPWs), so the number of rows per read is limited to
    keep reads under MAX_VALUES_PER_READ values.

    Calls process_block(time index, slice of rows, block) for every block read,
    with time index and slice relative to the ranges requested, and block a
    (rows, channels, values_per_channel) view. A block is released before the
    next one is read, so that only one read block is in memory at a time (which a
    generator could not ensure: the consumer keeps the previous block while the
    next one is read).
    """
    time_start, time_stop = time_range
    row_start, row_stop = row_range
    chan_start, chan_stop = frequency_range
    num_chan = chan_stop - chan_start
    segment_len = num_chan * values_per_channel
    rows_per_read = max(1, MAX_VALUES_PER_READ // max(row_len, 1))
    component_offset = None if isinstance(source, np.ndarray) else source.tell()

    for time_idx in range(time_start, time_stop):
        for first_row in range(row_start, row_stop, rows_per_read):
            last_row = min(first_row + rows_per_read, row_stop)
            start = (
                (time_idx * num_rows + first_row) * row_len
                + spw_offset
                + chan_start * values_per_channel
            )
            block = read_component_block(
                source,
                start,
                (last_row - first_row, segment_len),
                (row_len, 1),
                data_type=data_type,
                elements_count=elements_count,
                component_offset=component_offset,
            )
            process_block(
                time_idx - time_start,
                slice(first_row - row_start, last_row - row_start),
                block.reshape(last_row - first_row, num_chan, values_per_channel),
            )
            del block


def _range_len(index_range: tuple[int, int]) -> int:
    return index_range[1] - index_range[0]


def load_cross_data_one_spw(
    source: np.ndarray | typing.BinaryIO,
    layout: BDFSpwLayout,
    time_range: tuple[int, int],
    baseline_range: tuple[int, int],
    frequency_range: tuple[int, int],
    polarization_range: tuple[int, int],
    data_type: np.dtype | None = None,
    elements_count: int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Loads the visibilities of one SPW from the crossData binary component of one
    subset.

    Integer (INT16/INT32) data are divided by the scale factor of the SPW. FLOAT32
    data are used as they are. The APC given by the layout (AP_UNCORRECTED) is
    loaded. crossData are (re, im) pairs, except for data not from the CORRELATOR
    whose crossData size corresponds to real values (imaginary part 0 then).

    Parameters
    ----------
    source : np.ndarray | typing.BinaryIO
        crossData values (1D array from BDFReader.getSubset), or BDF file
        positioned at the beginning of the crossData binary component.
    layout : BDFSpwLayout
        Layout of the SPW data in the BDF.
    time_range : tuple[int, int]
        [start, stop) of the integrations (TIM samples) of the subset.
    baseline_range : tuple[int, int]
        [start, stop) of the cross baselines (crossData rows).
    frequency_range : tuple[int, int]
        [start, stop) of the channels.
    polarization_range : tuple[int, int]
        [start, stop) of the polarizations.
    data_type : np.dtype | None
        dtype of the values (when reading from a file).
    elements_count : int | None
        Number of values of the binary component (when reading from a file).
    out : np.ndarray | None
        complex64 array (time, baseline, frequency, polarization) of the selected
        shape to write into (can be a view of a larger array). None: a new array.

    Returns
    -------
    np.ndarray
        complex64 array (time, baseline, frequency, polarization): out, if given.
    """
    if isinstance(source, np.ndarray):
        data_type, elements_count = source.dtype, source.size

    # (re, im) pairs, or real values (accepted for data not from the CORRELATOR)
    values_per_pol = 2
    if layout.real_cross_data_allowed and elements_count == layout.cross_size // 2:
        values_per_pol = 1
    else:
        _check_component_size("crossData", elements_count, layout.cross_size)

    pol_start, pol_stop = polarization_range
    vis = output_array(
        out,
        (
            _range_len(time_range),
            _range_len(baseline_range),
            _range_len(frequency_range),
            pol_stop - pol_start,
        ),
        VISIBILITY_DTYPE,
    )
    if values_per_pol == 1:
        vis.imag[...] = 0
    scale_factor = (
        np.float64(layout.scale_factor) if np.dtype(data_type).kind in "iu" else None
    )

    def write_block(time_idx: int, rows: slice, block: np.ndarray):
        values = block.reshape(
            block.shape[:2] + (layout.num_cross_pols, values_per_pol)
        )
        values = values[..., pol_start:pol_stop, :]
        vis_block = vis[time_idx, rows]
        for idx, vis_part in enumerate(
            (vis_block.real, vis_block.imag)[:values_per_pol]
        ):
            if scale_factor is None:
                vis_part[...] = values[..., idx]
            else:
                np.divide(
                    values[..., idx],
                    scale_factor,
                    out=vis_part,
                    dtype=np.float64,
                    casting="unsafe",
                )

    _process_row_blocks(
        source,
        layout.num_cross_baselines,
        layout.cross_row_len * values_per_pol // 2,
        layout.cross_spw_offset * values_per_pol // 2,
        layout.num_cross_pols * values_per_pol,
        time_range,
        baseline_range,
        frequency_range,
        data_type,
        elements_count,
        write_block,
    )

    return vis


#: Decoding of full-polarization autoData (sdPolProducts XX XY YY, stored as the 4
#: floats XX, Re(XY), Im(XY), YY): for every output polarization, (index of the
#: real part, index of the imaginary part or None, sign of the imaginary part), by
#: number of output polarizations. autoData: "The choice of a real- vs.
#: complex-valued datum is dependent upon the polarization product...parallel-hand
#: polarizations are real-valued, while cross-hand polarizations are
#: complex-valued". With 4 output polarizations YX = conj(XY).
_FULL_POL_AUTO_DECODING = {
    3: ((0, None, 1), (1, 2, 1), (3, None, 1)),
    4: ((0, None, 1), (1, 2, 1), (1, 2, -1), (3, None, 1)),
}


def _decode_full_pol_auto_values(
    values: np.ndarray,
    num_polarizations: int,
    polarization_range: tuple[int, int],
    out: np.ndarray,
):
    """
    Writes full-polarization autoData values (..., channel, 4) into the complex64
    array out (..., channel, polarizations selected).
    """
    decoding = _FULL_POL_AUTO_DECODING[num_polarizations]
    for out_idx, (re_idx, im_idx, im_sign) in enumerate(
        decoding[slice(*polarization_range)]
    ):
        target = out[..., out_idx]
        target.real[...] = values[..., re_idx]
        if im_idx is None:
            target.imag[...] = 0
        elif im_sign > 0:
            target.imag[...] = values[..., im_idx]
        else:
            # Not np.negative(..., out=target.imag): with a strided input and a
            # strided output, numpy 2.5 (float32 SIMD loop) reads the input as if
            # it were contiguous. The temporary is the size of one read block.
            target.imag[...] = -values[..., im_idx]


def load_auto_data_one_spw(
    source: np.ndarray | typing.BinaryIO,
    layout: BDFSpwLayout,
    time_range: tuple[int, int],
    antenna_range: tuple[int, int],
    frequency_range: tuple[int, int],
    polarization_range: tuple[int, int],
    data_type: np.dtype | None = None,
    elements_count: int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Loads the auto-correlations of one SPW from the autoData binary component of
    one subset.

    Full-polarization autoData (sdPolProducts XX XY YY, stored as XX, Re(XY),
    Im(XY), YY) give [XX, XY, conj(XY), YY] when the output polarization axis has
    4 entries (CROSS_AND_AUTO data with 4 crossPolProducts), and [XX, XY, YY]
    otherwise (AUTO_ONLY). Real values get a zero imaginary part.

    Parameters
    ----------
    source : np.ndarray | typing.BinaryIO
        autoData values (1D array from BDFReader.getSubset), or BDF file
        positioned at the beginning of the autoData binary component.
    layout : BDFSpwLayout
        Layout of the SPW data in the BDF.
    time_range : tuple[int, int]
        [start, stop) of the integrations (TIM samples) of the subset.
    antenna_range : tuple[int, int]
        [start, stop) of the antennas (autoData rows).
    frequency_range : tuple[int, int]
        [start, stop) of the channels.
    polarization_range : tuple[int, int]
        [start, stop) of the (output) polarizations.
    data_type : np.dtype | None
        dtype of the values (when reading from a file).
    elements_count : int | None
        Number of values of the binary component (when reading from a file).
    out : np.ndarray | None
        complex64 array (time, antenna, frequency, polarization) of the selected
        shape to write into (can be a view of a larger array). None: a new array.

    Returns
    -------
    np.ndarray
        complex64 array (time, antenna, frequency, polarization): out, if given.
    """
    if isinstance(source, np.ndarray):
        data_type, elements_count = source.dtype, source.size
    _check_component_size("autoData", elements_count, layout.auto_size)

    pol_start, pol_stop = polarization_range
    vis = output_array(
        out,
        (
            _range_len(time_range),
            _range_len(antenna_range),
            _range_len(frequency_range),
            pol_stop - pol_start,
        ),
        VISIBILITY_DTYPE,
    )

    def write_block(time_idx: int, rows: slice, block: np.ndarray):
        if layout.num_sd_pols != 3:
            # real values: imaginary part 0
            vis[time_idx, rows] = block[..., pol_start:pol_stop]
        else:
            _decode_full_pol_auto_values(
                block,
                layout.num_polarizations,
                polarization_range,
                vis[time_idx, rows],
            )

    _process_row_blocks(
        source,
        layout.num_antennas,
        layout.auto_row_len,
        layout.auto_spw_offset,
        layout.num_auto_values,
        time_range,
        antenna_range,
        frequency_range,
        data_type,
        elements_count,
        write_block,
    )

    return vis
