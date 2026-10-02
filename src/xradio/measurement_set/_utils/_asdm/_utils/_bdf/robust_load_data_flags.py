"""
Module to load data and flags binary components from BDFs.
Robust in the sense that tolerates (common) inconsistencies in the BDF metadata,
including especially incomplete or incorrect axes definitions.

For the data it loads both autoData and crossData binary components for one SPW.

For both data and flags, re-arranges the values for the different times, baselines,
and polarizations in order to produce MSv4-style ndarrays with dimensions (time,
baseline, frequency, polarization). The baseline dimension holds the
cross-correlation baselines (in BDF order) followed by the auto-correlations (in
antenna order), or the antennas for AUTO_ONLY data.

Two levels:

- partition level (load_*_from_partition_bdfs): selection along the time axis of a
  partition/MSv4, made of the integrations of several BDFs. Any numpy "basic" index
  is accepted (ints, slices with steps, negative indices, None). The selection is
  loaded as a contiguous block (one step-1 slice with explicit bounds per dimension)
  and the rest of the index is applied to the block.
- BDF level (load_*_from_bdf): contiguous selection (step-1 slices) along the
  dimensions of the data of one SPW of one BDF, with BDF-local time (integration)
  indices.
"""

import time

import numpy as np
import pyasdm

from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
    apply_residual_keys,
    block_slice_len,
    find_bdfs_and_indices_in_selected_times,
    is_block_slice,
    split_dim_key,
)
from xradio.measurement_set._utils._asdm._utils._bdf.basebands_spws import (
    find_if_different_basebands_pols,
    find_if_different_basebands_spws,
    find_spw_in_basebands_list,
)
from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
    check_basebands,
    check_correlation_mode,
    check_data_components,
    check_num_bin,
    ensure_presence_binary_components,
    exclude_unsupported_axis_names,
    open_bdf,
    required_binary_components,
)
from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
    load_flags_all_subsets_from_trees,
    load_visibilities_all_subsets_from_trees,
)
from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_subset_array import (
    define_flag_shape,
    define_visibility_shape,
    load_flags_all_subsets,
    load_visibilities_all_subsets,
)

#: dtype of the visibilities produced (int16/int32 BDF data divided by the scale
#: factor, float32 data used as is, real-valued data with imaginary part 0)
VISIBILITY_DTYPE = np.dtype("complex64")
#: dtype of the flags produced (flagged = any bit of the BDF flag word set)
FLAG_DTYPE = np.dtype("bool")

_DIM_NAMES = ("time", "baseline", "frequency", "polarization")


def load_visibilities_from_partition_bdfs(
    bdf_paths: list[str],
    spw_id: int,
    time_indices_by_bdf: dict,
    array_slice: tuple[slice | int | None, ...] = (
        slice(None),
        slice(None),
        slice(None),
        slice(None),
    ),
) -> np.ndarray:
    """
    Load the visibilities of one SPW from the BDFs of a partition/MSv4.

    Parameters
    ----------
    bdf_paths : list[str]
        BDFs of the partition (in time order). Kept for backward compatibility:
        the BDFs read are time_indices_by_bdf["bdf_names"].
    spw_id : int
        Position of the SPW in the BDFs (over all basebands).
    time_indices_by_bdf : dict
        "bdf_names" (BDF paths) and "bdf_start" (index of the first integration of
        every BDF in the partition time axis, plus the total number of
        integrations), as produced by load_times_from_partition_bdfs.
    array_slice : tuple[slice | int | None, ...]
        Selection along (time, baseline, frequency, polarization): numpy basic
        indices (slices, ints, None for the whole dimension). Missing trailing
        dimensions are selected whole. Contiguous slices with explicit bounds
        (slice(start, stop[, 1]) with ints 0 <= start <= stop, as given by the
        xarray backend arrays) are not clipped: bounds beyond a dimension raise an
        error (this detects ASDM metadata / BDF mismatches).

    Returns
    -------
    np.ndarray
        complex64 visibilities with dimensions (time, baseline, frequency,
        polarization), minus the dimensions indexed with an int.

    Raises
    ------
    IndexError, ValueError, TypeError
        For selections out of range or not supported.
    NotImplementedError
        For data layouts that are not supported (numBin > 1, ...).
    RuntimeError
        If a BDF cannot be opened or its header parsed (BDFOpenError), if it is
        inconsistent with the selection or its loading fails. The message names
        the BDF.
    """
    return _load_from_partition_bdfs(
        spw_id,
        time_indices_by_bdf,
        array_slice,
        load_visibilities_from_bdf,
        VISIBILITY_DTYPE,
        "VISIBILITY",
    )


def load_flags_from_partition_bdfs(
    bdf_paths: list[str],
    spw_id: int,
    time_indices_by_bdf: dict,
    array_slice: tuple[slice | int | None, ...] = (
        slice(None),
        slice(None),
        slice(None),
        slice(None),
    ),
) -> np.ndarray:
    """
    Load the flags of one SPW from the BDFs of a partition/MSv4.

    Parameters
    ----------
    bdf_paths : list[str]
        BDFs of the partition (in time order). Kept for backward compatibility:
        the BDFs read are time_indices_by_bdf["bdf_names"].
    spw_id : int
        Position of the SPW in the BDFs (over all basebands).
    time_indices_by_bdf : dict
        "bdf_names" and "bdf_start", see load_visibilities_from_partition_bdfs.
    array_slice : tuple[slice | int | None, ...]
        Selection along (time, baseline, frequency, polarization), see
        load_visibilities_from_partition_bdfs.

    Returns
    -------
    np.ndarray
        Boolean flags (True when any bit of the BDF flag word is set) with
        dimensions (time, baseline, frequency, polarization), minus the dimensions
        indexed with an int. BDF flags are per (time, baseline, polarization), the
        same for all channels.

    Raises
    ------
    IndexError, ValueError, TypeError, NotImplementedError, RuntimeError
        See load_visibilities_from_partition_bdfs.
    """
    return _load_from_partition_bdfs(
        spw_id,
        time_indices_by_bdf,
        array_slice,
        load_flags_from_bdf,
        FLAG_DTYPE,
        "FLAG",
    )


def _complete_key(array_slice) -> tuple:
    """Turn array_slice into a tuple of 4 indices (None for a whole dimension)."""
    if array_slice is None:
        return (None,) * 4
    if not isinstance(array_slice, tuple | list):
        raise TypeError(
            f"Expected a tuple of indices (time, baseline, frequency, polarization), "
            f"got {type(array_slice)}: {array_slice!r}"
        )
    keys = tuple(array_slice)
    if len(keys) > 4:
        raise ValueError(f"Expected at most 4 indices, got {array_slice=}")
    return keys + (None,) * (4 - len(keys))


def _load_from_partition_bdfs(
    spw_id: int,
    time_indices_by_bdf: dict,
    array_slice,
    load_bdf_function,
    dtype: np.dtype,
    var_name: str,
) -> np.ndarray:
    start = time.perf_counter()

    keys = _complete_key(array_slice)
    bdf_names = list(time_indices_by_bdf["bdf_names"])
    bdf_start = list(time_indices_by_bdf["bdf_start"])
    if not bdf_names or len(bdf_start) != len(bdf_names) + 1:
        raise ValueError(
            f"Cannot load {var_name}: expected at least one BDF and len(bdf_start) "
            f"== len(bdf_names) + 1, got {time_indices_by_bdf=}"
        )
    time_len = int(bdf_start[-1])

    if all(is_block_slice(key) for key in keys):
        # contiguous block selection (the usual case when called from the xarray
        # backend arrays)
        if keys[0].stop > time_len:
            raise IndexError(
                f"Time selection {keys[0]} out of range for {time_len} integrations"
            )
        block_keys = tuple(slice(int(key.start), int(key.stop), 1) for key in keys)
        residuals = (None,) * 4
    else:
        dims = (time_len, *_partition_data_dims(bdf_names[0], spw_id))
        block_keys, residuals = zip(
            *(
                split_dim_key(key, dim_len)
                for key, dim_len in zip(keys, dims, strict=True)
            ),
            strict=True,
        )

    block_shape = tuple(block_slice_len(key) for key in block_keys)
    bdfs_in_selected_times = []
    if 0 in block_shape:
        block = np.zeros(block_shape, dtype=dtype)
    else:
        bdfs_in_selected_times, bdf_time_slices = (
            find_bdfs_and_indices_in_selected_times(time_indices_by_bdf, block_keys[0])
        )
        # When the selection is in one BDF (always the case for chunks of one
        # integration), the block loaded from the BDF is the result: no second
        # full-size array and copy. Otherwise the BDF blocks are copied into one
        # array (at most one BDF block alive besides it).
        single_bdf = len(bdfs_in_selected_times) == 1
        block = None if single_bdf else np.empty(block_shape, dtype=dtype)
        time_offset = 0
        for bdf_path, bdf_time_slice in zip(
            bdfs_in_selected_times, bdf_time_slices, strict=True
        ):
            bdf_key = (bdf_time_slice, *block_keys[1:])
            bdf_block = load_bdf_function(bdf_path, spw_id, bdf_key)
            bdf_shape = tuple(block_slice_len(key) for key in bdf_key)
            if bdf_block.shape != bdf_shape:
                raise RuntimeError(
                    f"Loaded {var_name} block with shape {bdf_block.shape} from BDF "
                    f"{bdf_path}, expected {bdf_shape} (selection {bdf_key})"
                )
            if single_bdf:
                block = _as_result_block(bdf_block, dtype)
            else:
                block[time_offset : time_offset + bdf_shape[0]] = bdf_block
            del bdf_block
            time_offset += bdf_shape[0]
        if time_offset != block_shape[0]:
            raise RuntimeError(
                f"Loaded {time_offset} integrations of {var_name} from BDFs "
                f"{bdfs_in_selected_times}, expected {block_shape[0]} for the time "
                f"selection {block_keys[0]} ({time_indices_by_bdf=})"
            )

    result = apply_residual_keys(block, residuals)

    end = time.perf_counter()
    xradio_logger().debug(
        f"Loaded {var_name} array, with shape {result.shape} from "
        f"{len(bdfs_in_selected_times)} BDFs (of {len(bdf_names)} in the "
        f"partition), time: {end - start:.6}"
    )

    return result


def _as_result_block(loaded: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    A block loaded from a BDF as the result of a partition-level load: the array
    itself when it is a writable, C-contiguous array of the result dtype that owns
    its memory (as produced by the BDF-level loaders), a copy otherwise (views,
    such as the broadcast BDF flags, which are read-only, or other dtypes).
    """
    if (
        loaded.dtype == dtype
        and loaded.flags.c_contiguous
        and loaded.flags.writeable
        and loaded.flags.owndata
    ):
        return loaded
    return np.array(loaded, dtype=dtype, order="C")


def make_bdf_description(bdf_header: pyasdm.bdf.BDFHeader) -> dict:
    """
    Make a dict with the information from a BDF header used by the loaders.

    Parameters
    ----------
    bdf_header : pyasdm.bdf.BDFHeader
        Header of a BDF.

    Returns
    -------
    dict
        BDF description. "binary_types" lists the binary components present in the
        BDF (non-zero size in the header); "sizes" and "axes" give their size
        (number of values per subset) and axes names.
    """
    binary_types = [
        binary_type
        for binary_type in dict.fromkeys(bdf_header.getBinaryTypes())
        if bdf_header.hasBinary(binary_type)
    ]
    bdf_descr = {
        # packed/TIM: dimensionality == 0
        "dimensionality": bdf_header.getDimensionality(),
        "num_time": bdf_header.getNumTime(),
        "processor_type": bdf_header.getProcessorType(),
        "binary_types": binary_types,
        "correlation_mode": bdf_header.getCorrelationMode(),
        "apc": list(bdf_header.getAPClist()),
        "num_antenna": bdf_header.getNumAntenna(),
        "basebands": bdf_header.getBasebandsList(),
        "sizes": {
            binary_type: bdf_header.getSize(binary_type) for binary_type in binary_types
        },
        "axes": {
            binary_type: list(bdf_header.getAxesNames(binary_type))
            for binary_type in binary_types
        },
    }

    return bdf_descr


def _bdf_data_dims(
    bdf_descr: dict, baseband_spw_idxs: tuple[int, int]
) -> tuple[int, int, int]:
    """
    Lengths of the (baseline, frequency, polarization) dimensions of the data and
    flags of one SPW of a BDF.
    """
    baseband_idx, spw_idx = baseband_spw_idxs
    spw_descr = bdf_descr["basebands"][baseband_idx]["spectralWindows"][spw_idx]
    num_antenna = bdf_descr["num_antenna"]
    if bdf_descr["correlation_mode"] == pyasdm.enumerations.CorrelationMode.AUTO_ONLY:
        baseline_len = num_antenna
        polarization_len = len(spw_descr["sdPolProducts"])
    else:
        baseline_len = num_antenna * (num_antenna - 1) // 2 + num_antenna
        polarization_len = len(spw_descr["crossPolProducts"]) or len(
            spw_descr["sdPolProducts"]
        )

    return baseline_len, spw_descr["numSpectralPoint"], polarization_len


def _partition_data_dims(bdf_path: str, spw_id: int) -> tuple[int, int, int]:
    """
    Lengths of the (baseline, frequency, polarization) dimensions of the data of an
    SPW, from the header of one BDF.
    """
    with open_bdf(bdf_path) as bdf_reader:
        bdf_descr = make_bdf_description(bdf_reader.getHeader())
    baseband_spw_idxs = find_spw_in_basebands_list(
        spw_id, bdf_descr["basebands"], bdf_path
    )
    return _bdf_data_dims(bdf_descr, baseband_spw_idxs)


def _normalize_bdf_key(
    array_slice,
    bdf_descr: dict,
    baseband_spw_idxs: tuple[int, int],
    bdf_path: str,
) -> tuple[slice, slice, slice, slice]:
    """
    Turn a BDF-level selection into 4 block slices slice(start, stop, 1) with
    explicit bounds, checking them against the dimensions of the BDF.

    The time selection (BDF-local integration indices) must have explicit bounds,
    except for packed BDFs (dimensionality 0), whose number of integrations
    (numTime) is known from the header. Other dimensions accept None or
    slice(None) for the whole dimension.
    """
    keys = _complete_key(array_slice)
    dims = _bdf_data_dims(bdf_descr, baseband_spw_idxs)

    time_key = keys[0]
    num_time = bdf_descr["num_time"] if bdf_descr["dimensionality"] == 0 else None
    if time_key is None or time_key == slice(None):
        if num_time is None:
            raise ValueError(
                f"The time selection for BDF {bdf_path} needs explicit bounds "
                f"(slice(start, stop)), got {time_key}"
            )
        time_key = slice(0, num_time, 1)
    if not is_block_slice(time_key):
        raise ValueError(
            f"The time selection for BDF {bdf_path} must be a slice(start, stop) "
            f"with explicit non-negative bounds and step 1, got {time_key}"
        )
    if num_time is not None and time_key.stop > num_time:
        raise RuntimeError(
            f"Time selection {time_key} out of range for packed BDF {bdf_path} with "
            f"numTime={num_time}"
        )
    block_keys = [slice(int(time_key.start), int(time_key.stop), 1)]

    for key, dim_len, dim_name in zip(keys[1:], dims, _DIM_NAMES[1:], strict=True):
        if key is None or key == slice(None):
            block_keys.append(slice(0, dim_len, 1))
            continue
        if not is_block_slice(key):
            raise ValueError(
                f"The {dim_name} selection for BDF {bdf_path} must be a "
                f"slice(start, stop) with explicit non-negative bounds and step 1, "
                f"got {key}"
            )
        if key.stop > dim_len:
            raise RuntimeError(
                f"The {dim_name} selection {key} exceeds the {dim_len} {dim_name} "
                f"elements of SPW {baseband_spw_idxs[1]} of baseband "
                f"{baseband_spw_idxs[0]} in BDF {bdf_path}"
            )
        block_keys.append(slice(int(key.start), int(key.stop), 1))

    return tuple(block_keys)


def _check_loaded_block(
    loaded: np.ndarray | None,
    expected_shape: tuple[int, ...],
    what: str,
    bdf_path: str,
) -> np.ndarray:
    if loaded is None:
        raise RuntimeError(f"Could not load {what} from BDF {bdf_path}")
    loaded = np.asarray(loaded)
    if loaded.shape != expected_shape:
        raise RuntimeError(
            f"Loaded {what} from BDF {bdf_path} with shape {loaded.shape}, expected "
            f"{expected_shape}"
        )
    return loaded


def load_visibilities_from_bdf(
    bdf_path: str,
    spw_id: int,
    array_slice: tuple[slice, ...],
    never_reshape_from_all_spws: bool = True,
) -> np.ndarray:
    """
    Load a block of visibilities of one SPW from one BDF.

    Parameters
    ----------
    bdf_path : str
        Path of the BDF.
    spw_id : int
        Position of the SPW in the BDF (over all basebands).
    array_slice : tuple[slice, ...]
        Selection along (time, baseline, frequency, polarization) as step-1 slices.
        The time selection holds BDF-local integration indices and needs explicit
        bounds (except for packed BDFs). The other dimensions accept None or
        slice(None) for the whole dimension.
    never_reshape_from_all_spws : bool
        Use the loader that looks for the SPW in the data trees (works for any
        BDF), instead of the reshape-based loader (only for BDFs with the same
        number of SPWs per baseband and of channels per SPW).

    Returns
    -------
    np.ndarray
        complex64 visibilities with dimensions (time, baseline, frequency,
        polarization).

    Raises
    ------
    NotImplementedError
        For data layouts that are not supported (numBin > 1, ...).
    RuntimeError
        If the BDF cannot be opened or its header parsed (BDFOpenError), if it is
        inconsistent with the selection or its loading fails. The message names
        the BDF.
    """
    with open_bdf(bdf_path) as bdf_reader:
        bdf_header = bdf_reader.getHeader()
        bdf_descr = make_bdf_description(bdf_header)

        check_correlation_mode(bdf_descr["correlation_mode"], bdf_path)
        check_basebands(bdf_descr["basebands"], bdf_path)
        ensure_presence_binary_components(
            required_binary_components(bdf_descr["correlation_mode"]),
            bdf_descr["binary_types"],
            bdf_path,
        )

        baseband_spw_idxs = find_spw_in_basebands_list(
            spw_id, bdf_descr["basebands"], bdf_path
        )
        bdf_descr["apc_index"] = check_data_components(
            bdf_descr, baseband_spw_idxs, bdf_path
        )
        bdf_key = _normalize_bdf_key(
            array_slice, bdf_descr, baseband_spw_idxs, bdf_path
        )
        expected_shape = tuple(block_slice_len(key) for key in bdf_key)
        if 0 in expected_shape:
            return np.zeros(expected_shape, dtype=VISIBILITY_DTYPE)

        different_channels_per_spw = find_if_different_basebands_spws(
            bdf_descr["basebands"]
        )
        guessed_shape = define_visibility_shape(bdf_descr, baseband_spw_idxs)
        try:
            if never_reshape_from_all_spws or different_channels_per_spw:
                bdf_vis = load_visibilities_all_subsets_from_trees(
                    bdf_reader, guessed_shape, baseband_spw_idxs, bdf_descr, bdf_key
                )
            else:
                bdf_vis = load_visibilities_all_subsets(
                    bdf_reader, guessed_shape, baseband_spw_idxs, bdf_descr, bdf_key
                )
        except NotImplementedError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Error while loading data/visibilities from a BDF ({bdf_path=}, "
                f"selection {bdf_key}). Details: {exc!r}\nBDF header:\n{bdf_header}"
            ) from exc

        bdf_vis = _check_loaded_block(bdf_vis, expected_shape, "visibilities", bdf_path)

    return bdf_vis.astype(VISIBILITY_DTYPE, copy=False)


def check_flags_dims(flags_dims: list[str], bdf_path: str | None = None):
    """
    Check that the axes of the flags binary component are supported.

    Parameters
    ----------
    flags_dims : list[str]
        Axes names of the flags binary component.
    bdf_path : str | None
        Path of the BDF (for error messages).

    Raises
    ------
    RuntimeError
        If unsupported axes are found.
    """
    exclude_unsupported_axis_names(flags_dims, True, bdf_path)


def load_flags_from_bdf(
    bdf_path: str,
    spw_id: int,
    array_slice: tuple[slice, ...],
    never_reshape_from_all_spws: bool = False,
) -> np.ndarray:
    """
    Load a block of flags of one SPW from one BDF.

    Parameters
    ----------
    bdf_path : str
        Path of the BDF.
    spw_id : int
        Position of the SPW in the BDF (over all basebands).
    array_slice : tuple[slice, ...]
        Selection along (time, baseline, frequency, polarization), see
        load_visibilities_from_bdf.
    never_reshape_from_all_spws : bool
        Use the loader that looks for the SPW in the flags trees, instead of the
        reshape-based loader (only for BDFs with the same polarizations in every
        SPW).

    Returns
    -------
    np.ndarray
        Boolean flags with dimensions (time, baseline, frequency, polarization).
        All False when the BDF has no flags binary component.

    Raises
    ------
    NotImplementedError
        For data layouts that are not supported (numBin > 1).
    RuntimeError
        If the BDF cannot be opened or its header parsed (BDFOpenError), if it is
        inconsistent with the selection or its loading fails. The message names
        the BDF.
    """
    with open_bdf(bdf_path) as bdf_reader:
        bdf_header = bdf_reader.getHeader()

        check_flags_dims(bdf_header.getAxesNames("flags"), bdf_path)

        bdf_descr = make_bdf_description(bdf_header)

        check_correlation_mode(bdf_descr["correlation_mode"], bdf_path)
        check_basebands(bdf_descr["basebands"], bdf_path)

        baseband_spw_idxs = find_spw_in_basebands_list(
            spw_id, bdf_descr["basebands"], bdf_path
        )
        check_num_bin(bdf_descr["basebands"], baseband_spw_idxs, bdf_path)
        bdf_key = _normalize_bdf_key(
            array_slice, bdf_descr, baseband_spw_idxs, bdf_path
        )
        expected_shape = tuple(block_slice_len(key) for key in bdf_key)
        if 0 in expected_shape or "flags" not in bdf_descr["binary_types"]:
            # No flags binary component in the BDF: nothing is flagged
            return np.zeros(expected_shape, dtype=FLAG_DTYPE)

        different_pols_per_spw = find_if_different_basebands_pols(
            bdf_descr["basebands"]
        )
        guessed_shape = define_flag_shape(bdf_descr, baseband_spw_idxs)
        try:
            if never_reshape_from_all_spws or different_pols_per_spw:
                bdf_flag = load_flags_all_subsets_from_trees(
                    bdf_reader,
                    guessed_shape,
                    bdf_descr,
                    baseband_spw_idxs,
                    bdf_key,
                )
            else:
                bdf_flag = load_flags_all_subsets(
                    bdf_reader, guessed_shape, baseband_spw_idxs, bdf_key
                )
            if bdf_flag is None:
                raise RuntimeError("the flags loader returned no flags")
            bdf_flag = _expand_frequency_in_flags_subset(
                np.asarray(bdf_flag),
                bdf_descr,
                baseband_spw_idxs[0],
                baseband_spw_idxs[1],
                bdf_key,
            )
        except NotImplementedError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Error while loading flags from a BDF ({bdf_path=}, selection "
                f"{bdf_key}). Details: {exc!r}\nBDF header:\n{bdf_header}"
            ) from exc

        bdf_flag = _check_loaded_block(bdf_flag, expected_shape, "flags", bdf_path)

    if bdf_flag.dtype != FLAG_DTYPE:
        # flagged = any bit of the flag word set
        bdf_flag = bdf_flag != 0

    return bdf_flag


def _expand_frequency_in_flags_subset(
    flag_subset: np.ndarray,
    bdf_descr: dict,
    baseband_idx: int,
    spw_idx: int,
    array_slice: tuple[slice | int | None, ...] | None,
) -> np.ndarray:
    """
    The BDF flags binary components do not give flags per-channel. Expand per-SPW
    flags from a BDF, with dimensions (time, baseline, polarization), into MSv4
    flags with dimensions (time, baseline, frequency, polarization), with as many
    channels as selected by array_slice[2].

    Parameters
    ----------
    flag_subset : np.ndarray
        Flags with dimensions (time, baseline, polarization). Flags that already
        have a frequency dimension (of length 1 or of the selected length) are also
        accepted.
    bdf_descr : dict
        BDF description (gives the number of channels of the SPW).
    baseband_idx : int
        Index of the baseband of the SPW.
    spw_idx : int
        Index of the SPW within the baseband.
    array_slice : tuple[slice | int | None, ...] | None
        Selection along (time, baseline, frequency, polarization). Only the
        frequency index is used: None or a missing index selects all channels, an
        int selects one channel and drops the frequency dimension.

    Returns
    -------
    np.ndarray
        Flags with dimensions (time, baseline, frequency, polarization), or
        (time, baseline, polarization) for an int frequency index. The expansion
        is a broadcast (read-only) view.

    Raises
    ------
    ValueError
        If the flags do not have the expected dimensions.
    """
    frequency_key = (
        array_slice[2] if array_slice is not None and len(array_slice) >= 3 else None
    )
    num_chan = bdf_descr["basebands"][baseband_idx]["spectralWindows"][spw_idx][
        "numSpectralPoint"
    ]

    if isinstance(frequency_key, int | np.integer):
        if flag_subset.ndim != 3:
            raise ValueError(
                f"Expected flags with dimensions (time, baseline, polarization), got "
                f"shape {flag_subset.shape}"
            )
        return flag_subset

    frequency_block, frequency_residual = split_dim_key(frequency_key, num_chan)
    frequency_len = (
        block_slice_len(frequency_block)
        if frequency_residual is None
        else len(frequency_residual)
    )

    if flag_subset.ndim == 3:
        expanded_shape = (*flag_subset.shape[0:2], frequency_len, flag_subset.shape[2])
        return np.broadcast_to(flag_subset[:, :, np.newaxis, :], expanded_shape)

    if flag_subset.ndim == 4 and flag_subset.shape[2] in (1, frequency_len):
        expanded_shape = (*flag_subset.shape[0:2], frequency_len, flag_subset.shape[3])
        return np.broadcast_to(flag_subset, expanded_shape)

    raise ValueError(
        f"Expected flags with dimensions (time, baseline, polarization), got shape "
        f"{flag_subset.shape} (with {frequency_len} channels selected)"
    )
