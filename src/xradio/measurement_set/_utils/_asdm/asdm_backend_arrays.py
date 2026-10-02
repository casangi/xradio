"""
Lazily indexable backend arrays of the ASDM xarray backend.

All the arrays derive from :class:`ASDMBackendArray`, which normalises the keys
produced by xarray (integers, slices with any step, empty selections) so that the
loaders only ever see one ``slice(start, stop, 1)`` per dimension, with Python
ints and ``0 <= start < stop <= dim_len``. The loaders return the complete block
(keeping all dimensions) and the wrapper applies the rest of the key (integer
squeezes, steps, reversals), casts to the declared dtype and checks the shape.
"""

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike

from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
    load_flags_from_partition_bdfs,
    load_visibilities_from_partition_bdfs,
)
from xradio.measurement_set._utils._asdm._utils.calculate_uvw import (
    calculate_uvw,
    check_uvw_inputs,
)


def normalize_basic_key(
    key: tuple, shape: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple, tuple[int, ...]]:
    """
    Split a basic (integers and slices) key into a step-1 block key and the key
    to apply to that block.

    Parameters
    ----------
    key : tuple
        Basic indexing key: one int or slice per dimension (missing trailing
        dimensions are selected entirely).
    shape : tuple[int, ...]
        Shape of the indexed array.

    Returns
    -------
    tuple[tuple[slice, ...], tuple, tuple[int, ...]]
        - block_key: one ``slice(start, stop, 1)`` per dimension, with Python ints
          and ``0 <= start <= stop <= dim_len``: the bounding block of the
          selection (integers become length-1 ranges). ``start == stop`` (only)
          for empty selections.
        - residual_key: key that applied to the block gives the selection
          (``0`` for integer keys, slices for the steps / reversals).
        - result_shape: shape of the selection (integer-indexed dimensions are
          dropped).

    Raises
    ------
    IndexError
        If there are too many indices or an integer index is out of bounds.
    TypeError
        If an index is not an int or a slice.
    """
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) > len(shape):
        raise IndexError(
            f"Too many indices ({len(key)}) for an array with {len(shape)} dimensions"
        )
    key = key + (slice(None),) * (len(shape) - len(key))

    block_key = []
    residual_key = []
    result_shape = []
    for dim_key, dim_len in zip(key, shape, strict=True):
        if isinstance(dim_key, int | np.integer) and not isinstance(
            dim_key, bool | np.bool_
        ):
            index = int(dim_key)
            if not -dim_len <= index < dim_len:
                raise IndexError(
                    f"Index {index} is out of bounds for a dimension of size {dim_len}"
                )
            index %= dim_len
            block_key.append(slice(index, index + 1, 1))
            residual_key.append(0)
        elif isinstance(dim_key, slice):
            selected = range(dim_len)[dim_key]
            result_shape.append(len(selected))
            if len(selected) == 0:
                block_key.append(slice(0, 0, 1))
                residual_key.append(slice(None))
                continue
            first, last = selected[0], selected[-1]
            start, stop = min(first, last), max(first, last) + 1
            block_key.append(slice(start, stop, 1))
            if selected.step == 1:
                residual_key.append(slice(None))
            elif selected.step > 0:
                residual_key.append(slice(0, None, selected.step))
            else:
                residual_key.append(slice(stop - start - 1, None, selected.step))
        else:
            raise TypeError(
                f"Unsupported basic index {dim_key!r} of type {type(dim_key)}: only "
                "integers and slices are supported"
            )

    return tuple(block_key), tuple(residual_key), tuple(result_shape)


def _replace_empty_slices(
    key: xr.core.indexing.ExplicitIndexer, shape: tuple[int, ...]
) -> xr.core.indexing.ExplicitIndexer:
    """
    Replace slices that select nothing by ``slice(0, 0)``. xarray's key
    decomposition fails (IndexError) on empty slices with a negative step.
    """
    empty = [
        isinstance(dim_key, slice) and len(range(dim_len)[dim_key]) == 0
        for dim_key, dim_len in zip(key.tuple, shape, strict=True)
    ]
    if not any(empty):
        return key
    return type(key)(
        tuple(
            slice(0, 0) if is_empty else dim_key
            for dim_key, is_empty in zip(key.tuple, empty, strict=True)
        )
    )


class ASDMBackendArray(xr.backends.BackendArray):
    """
    Base class of the lazily indexable ASDM backend arrays.

    Subclasses implement :meth:`_raw_indexing_method`, which receives a normalised
    key (see :func:`normalize_basic_key`): a tuple with one
    ``slice(start, stop, 1)`` per dimension, with Python ints and
    ``0 <= start < stop <= dim_len`` (never None, never empty, never an int,
    never a step != 1). It must return an array with exactly ``stop - start``
    elements along every dimension (all dimensions kept).

    The wrapper (:meth:`__getitem__`) handles integer keys (squeeze), steps and
    negative steps (applied to the loaded block), empty selections (no loader
    call), casts the result to the declared dtype and checks its shape.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the array.
    dtype : DTypeLike
        Declared dtype. The arrays returned by indexing always have this dtype.
    """

    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike):
        shape = tuple(int(dim_len) for dim_len in shape)
        if any(dim_len < 0 for dim_len in shape):
            raise ValueError(f"Invalid (negative) array shape: {shape}")
        self._shape = shape
        self._dtype = np.dtype(dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, key: xr.core.indexing.ExplicitIndexer) -> np.ndarray:
        """
        Makes the ASDM backend arrays indexable (subscriptable with []), via the
        LazilyIndexedArray wrapper class.

        'key' is an explicit indexer from xarray. Outer / vectorized indexers are
        decomposed by xarray into a basic indexer (handled by
        :meth:`_getitem_basic`) and a NumPy indexer applied to its result.
        """
        return xr.core.indexing.explicit_indexing_adapter(
            _replace_empty_slices(key, self.shape),
            self.shape,
            xr.core.indexing.IndexingSupport.BASIC,
            self._getitem_basic,
        )

    def _getitem_basic(self, key: tuple) -> np.ndarray:
        """Index with a basic key (ints and slices), see the class docstring."""
        block_key, residual_key, result_shape = normalize_basic_key(key, self.shape)

        block_shape = tuple(dim_key.stop - dim_key.start for dim_key in block_key)
        if 0 in block_shape:
            return np.empty(result_shape, dtype=self.dtype)

        try:
            block = np.asarray(self._raw_indexing_method(block_key))
        except Exception as exc:
            message = str(exc).strip()
            summary = message.splitlines()[0][:300] if message else ""
            xradio_logger().warning(
                f"Exception while loading {type(self).__name__} block {block_key=} "
                f"(for {key=}): {type(exc).__name__}: {summary}"
            )
            raise

        if block.shape != block_shape:
            raise RuntimeError(
                f"{type(self).__name__}: the loader returned an array of shape "
                f"{block.shape} for the block {block_key}, expected {block_shape}"
            )

        trivial_residual = all(dim_key == slice(None) for dim_key in residual_key)
        result = block if trivial_residual else block[residual_key]
        result = self._cast(result, copy=not trivial_residual)

        if result.shape != result_shape:
            raise RuntimeError(
                f"{type(self).__name__}: indexing with {key=} produced shape "
                f"{result.shape}, expected {result_shape}"
            )
        return result

    def _cast(self, values: np.ndarray, copy: bool) -> np.ndarray:
        """
        Cast to the declared dtype. Returns a writeable array that does not keep
        a larger loaded block alive (copy when values is a view into it).
        """
        if np.iscomplexobj(values) and self.dtype.kind != "c":
            raise TypeError(
                f"{type(self).__name__}: the loader returned complex values "
                f"({values.dtype}) for an array declared as {self.dtype}"
            )
        if copy or values.dtype != self.dtype or not values.flags.writeable:
            return np.array(values, dtype=self.dtype)
        return values

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        """Load the block given by a normalised key (one step-1 slice per dim)."""
        raise NotImplementedError


class VisibilityArray(ASDMBackendArray):
    """
    For the MSv4 VISIBILITY data var, dims (time, baseline_id, frequency,
    polarization).

    Parameters
    ----------
    shape : tuple[int, ...]
        (time, baseline_id, frequency, polarization) shape.
    bdf_paths : list[str]
        Paths of the BDFs of the partition (ordered by time).
    bdf_spw_id : int
        Index of the SPW in the BDFs.
    time_indices_by_bdf : dict
        {"bdf_names": [...], "bdf_start": [...]} time indices of every BDF.
    dtype : DTypeLike, optional
        Complex dtype of the array, by default complex64.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        bdf_paths: list[str],
        bdf_spw_id: int,
        time_indices_by_bdf: dict,
        dtype: DTypeLike = np.complex64,
    ):
        super().__init__(shape, dtype)
        self._check_dtype_kind()
        self._bdf_paths = bdf_paths
        self._bdf_spw_id = bdf_spw_id
        self._time_indices_by_bdf = time_indices_by_bdf

    def _check_dtype_kind(self):
        if self.dtype.kind != "c":
            raise ValueError(
                f"{type(self).__name__} needs a complex dtype, got {self.dtype}"
            )

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        xradio_logger().debug(f" {type(self).__name__}._raw_indexing_method, {key=}")
        return load_visibilities_from_partition_bdfs(
            self._bdf_paths, self._bdf_spw_id, self._time_indices_by_bdf, key
        )


class SpectrumArray(VisibilityArray):
    """
    For the MSv4 SPECTRUM data var (single dish), dims (time, antenna_name,
    frequency, polarization): the real part of the auto-correlation data of
    AUTO_ONLY partitions.

    Parameters
    ----------
    shape : tuple[int, ...]
        (time, antenna_name, frequency, polarization) shape.
    bdf_paths : list[str]
        Paths of the BDFs of the partition (ordered by time).
    bdf_spw_id : int
        Index of the SPW in the BDFs.
    time_indices_by_bdf : dict
        {"bdf_names": [...], "bdf_start": [...]} time indices of every BDF.
    dtype : DTypeLike, optional
        Real floating point dtype of the array, by default float32.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        bdf_paths: list[str],
        bdf_spw_id: int,
        time_indices_by_bdf: dict,
        dtype: DTypeLike = np.float32,
    ):
        super().__init__(shape, bdf_paths, bdf_spw_id, time_indices_by_bdf, dtype)

    def _check_dtype_kind(self):
        if self.dtype.kind != "f":
            raise ValueError(
                f"{type(self).__name__} needs a real floating point dtype, got "
                f"{self.dtype}"
            )

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        return np.ascontiguousarray(np.real(super()._raw_indexing_method(key)))


class WeightArray(ASDMBackendArray):
    """
    For the MSv4 WEIGHT data var: constant 1.0 (the ASDM BDFs have no weights).
    Only the selected elements are ever materialised.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the array.
    dtype : DTypeLike, optional
        Floating point dtype, by default float64.
    """

    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike = np.float64):
        super().__init__(shape, dtype)

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        xradio_logger().debug(f" WeightArray._raw_indexing_method, {key=}")
        block_shape = tuple(dim_key.stop - dim_key.start for dim_key in key)
        # Read-only broadcast view (no memory): the wrapper copies only the
        # selected elements
        return np.broadcast_to(np.ones((), dtype=self.dtype), block_shape)


class PerTimeArray(ASDMBackendArray):
    """
    For the MSv4 data vars with one value per integration, the same for all
    the baselines (or antennas), dims (time, baseline_id or antenna_name):
    TIME_CENTROID and EFFECTIVE_INTEGRATION_TIME. Only the per-integration
    values are stored: the selected elements are materialised (as writeable
    arrays, like the other data variables) when they are accessed.

    Parameters
    ----------
    values : np.ndarray
        One value per integration (1-D, the length of the time dimension).
    num_rows : int
        Length of the second dimension (number of baselines or antennas).
    dtype : DTypeLike, optional
        Floating point dtype, by default float64.
    """

    def __init__(
        self, values: np.ndarray, num_rows: int, dtype: DTypeLike = np.float64
    ):
        # own read-only copy: later changes to the input do not alter the array
        values = np.array(values, dtype=dtype)
        if values.ndim != 1:
            raise ValueError(
                f"{type(self).__name__} needs one value per integration (1-D), got "
                f"values with shape {values.shape}"
            )
        super().__init__((len(values), num_rows), dtype)
        values.flags.writeable = False
        self._values = values

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        time_key, row_key = key
        # Read-only broadcast view (no memory per row): the wrapper copies it
        # into a writeable array of the selected elements only
        return np.broadcast_to(
            self._values[time_key, np.newaxis],
            (time_key.stop - time_key.start, row_key.stop - row_key.start),
        )


class FlagArray(ASDMBackendArray):
    """
    For the MSv4 FLAG data var (bool), dims (time, baseline_id or antenna_name,
    frequency, polarization).

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the array.
    bdf_paths : list[str]
        Paths of the BDFs of the partition (ordered by time).
    bdf_spw_id : int
        Index of the SPW in the BDFs.
    time_indices_by_bdf : dict
        {"bdf_names": [...], "bdf_start": [...]} time indices of every BDF.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        bdf_paths: list[str],
        bdf_spw_id: int,
        time_indices_by_bdf: dict,
    ):
        super().__init__(shape, np.dtype("bool"))
        self._bdf_paths = bdf_paths
        self._bdf_spw_id = bdf_spw_id
        self._time_indices_by_bdf = time_indices_by_bdf

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        xradio_logger().debug(f" FlagArray._raw_indexing_method, {key=}")
        flags = np.asarray(
            load_flags_from_partition_bdfs(
                self._bdf_paths, self._bdf_spw_id, self._time_indices_by_bdf, key
            )
        )
        # K2: flagged = any bit of the BDF flag word set
        return flags != 0 if flags.dtype != bool else flags


class UVWArray(ASDMBackendArray):
    """
    For the MSv4 UVW data var, dims (time, baseline_id, uvw_label), calculated on
    demand for the selected times and baselines (see
    :func:`~xradio.measurement_set._utils._asdm._utils.calculate_uvw.calculate_uvw`).

    The inputs are checked when the array is created, that is when the partition
    is opened (see
    :func:`~xradio.measurement_set._utils._asdm._utils.calculate_uvw.check_uvw_inputs`):
    a partition whose UVW cannot be calculated (for example, with a phase center
    in an unsupported frame) then fails to open, and open_asdm skips it (K13),
    instead of failing later in a compute of the processing set.

    Parameters
    ----------
    shape : tuple[int, ...]
        (time, baseline_id, uvw_label) shape.
    time : xr.DataArray
        Time measure (with units/format/scale attributes) of the time dim.
    baseline_antenna1_name : xr.DataArray
        Name of the first antenna of every baseline.
    baseline_antenna2_name : xr.DataArray
        Name of the second antenna of every baseline.
    antenna_position : xr.DataArray
        ITRS antenna positions, dims (antenna_name, cartesian label).
    phase_center_direction : xr.DataArray
        Phase center direction, dims (time, sky_dir_label) (one per time) or
        (field_name, sky_dir_label) with one field_name.

    Raises
    ------
    NotImplementedError
        If the phase center frame is not supported by the UVW calculation.
    ValueError
        If the inputs cannot give the UVW of this shape (missing time attributes,
        unknown baseline antennas, inconsistent sizes, ...).
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        time: xr.DataArray,
        baseline_antenna1_name: xr.DataArray,
        baseline_antenna2_name: xr.DataArray,
        antenna_position: xr.DataArray,
        phase_center_direction: xr.DataArray,
    ):
        super().__init__(shape, np.dtype("float64"))
        if len(self.shape) != 3:
            raise ValueError(
                f"UVWArray needs a (time, baseline_id, uvw_label) shape, got {shape}"
            )
        check_uvw_inputs(
            time,
            baseline_antenna1_name,
            baseline_antenna2_name,
            antenna_position,
            phase_center_direction,
        )
        expected_shape = (time.sizes["time"], np.size(baseline_antenna1_name), 3)
        if self.shape != expected_shape:
            raise ValueError(
                f"UVWArray shape {self.shape} does not match the inputs: expected "
                f"{expected_shape} (num_time, num_baseline, 3 uvw_label)"
            )
        self.time = time
        self.baseline_antenna1_name = baseline_antenna1_name
        self.baseline_antenna2_name = baseline_antenna2_name
        self.antenna_position = antenna_position
        self.phase_center_direction = phase_center_direction

    def _raw_indexing_method(self, key: tuple[slice, ...]) -> np.ndarray:
        xradio_logger().debug(f" UVWArray._raw_indexing_method, {key=}")
        return calculate_uvw(
            key,
            self.time,
            self.baseline_antenna1_name,
            self.baseline_antenna2_name,
            self.antenna_position,
            self.phase_center_direction,
        )
