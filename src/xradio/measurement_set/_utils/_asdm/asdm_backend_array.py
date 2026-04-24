import numpy as np

import xarray as xr


from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
    load_visibilities_from_partition_bdfs,
    load_flags_from_partition_bdfs,
)


class ASDMBackendArray(xr.backends.BackendArray):
    """To support lazy loading and indexing in the Xarray Backend for
    ASDM data."""

    def __init__(self, shape, dtype):
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self) -> tuple[int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, key: xr.core.indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        """'key' is a tuple of slices/integers provided by Xarray's indexer."""
        return xr.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr.core.indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        raise NotImplementedError


class VisibilityArray(ASDMBackendArray):
    """For the MSv4 VISIBILITY data var"""

    def __init__(self, shape: tuple[int], bdf_paths: list[str], bdf_spw_id: int):
        super().__init__(shape, np.dtype("complex128"))
        self._bdf_paths = bdf_paths
        self._bdf_spw_id = bdf_spw_id

    def _raw_indexing_method(self, key: tuple):
        xradio_logger().debug(f" VisibilityArray._raw_indexing_method, {key=}")
        visibility = load_visibilities_from_partition_bdfs(
            self._bdf_paths, self._bdf_spw_id, key
        )
        return visibility


class WeightArray(ASDMBackendArray):
    """For the MSv4 WEIGHT data var"""

    def __init__(self, shape):
        super().__init__(shape, np.dtype("float64"))

    def _raw_indexing_method(self, key: tuple):
        xradio_logger().debug(f" WeightArray._raw_indexing_method, {key=}")
        weight = np.ones(shape=self.shape, dtype=self.dtype)
        weight = weight[key]
        return weight


class FlagArray(ASDMBackendArray):
    """For the MSv4 FLAG data var"""

    def __init__(self, shape: tuple[int], bdf_paths: list[str], bdf_spw_id: int):
        super().__init__(shape, np.dtype("bool"))
        self._bdf_paths = bdf_paths
        self._bdf_spw_id = bdf_spw_id

    def _raw_indexing_method(self, key: tuple):
        xradio_logger().debug(f" FlagArray._raw_indexing_method, {key=}")
        flags = load_flags_from_partition_bdfs(self._bdf_paths, self._bdf_spw_id, key)
        return flags


class UVWArray(ASDMBackendArray):
    """For the MSv4 UVW data var"""

    def __init__(self, shape):
        super().__init__(shape, np.dtype("float64"))

    def _raw_indexing_method(self, key: tuple):
        xradio_logger().debug(f" UVWArray._raw_indexing_method, {key=}")
        return np.ones(shape=self.shape, dtype=self.dtype)
