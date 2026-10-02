"""
Xarray backend entrypoints for opening CASA and FITS images as xradio image
datasets, see https://docs.xarray.dev/en/latest/api/backends.html.

With xradio installed, images can be opened directly with xarray::

    import xarray as xr

    xds = xr.open_dataset("my_image.im", engine="xradio_casa_image")
    xds = xr.open_dataset("my_image.fits", engine="xradio_fits_image")

The backends are registered under the ``xradio_casa_image`` and
``xradio_fits_image`` engine names through the ``xarray.backends`` entry
points in pyproject.toml, which point at the light top level module
``_xradio_xarray_backends`` of the distribution (re-exported here), so that
xarray loads them without importing xradio. Without an explicit engine,
xarray picks ``xradio_casa_image`` for casacore image tables and
``xradio_fits_image`` for files ending in ``.fits``, ``.fit`` or ``.fts`` whose
primary HDU holds an image. The returned datasets conform to the image schema
(:py:class:`xradio.image.schema.ImageXds`).

Data variables are read lazily, like those of xarray's own backends: opening
reads only metadata, ``ds.load()`` (or ``.values``) reads the pixels, and
indexing reads only the image planes needed. Pass ``chunks`` to
``xr.open_dataset`` for dask-backed variables: every dask chunk then reads only
its own planes. The images are read in blocks of one plane (one time,
frequency and polarization) by default, which the variables advertise as
``encoding["preferred_chunks"]``, so ``chunks={}`` gives one dask chunk per
plane and ``chunks="auto"`` groups whole planes. Each block read has a fixed
cost, so for cubes with many small planes pass larger blocks, e.g.
``image_chunks={"frequency": 10}``. The beam fit parameters, which come from
the image metadata, are loaded when the image is opened.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping

import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

from _xradio_xarray_backends import (
    CasaImageBackendEntrypoint,
    FitsImageBackendEntrypoint,
)
from xradio._utils.xarray_helpers import (
    remove_variable_references,
    remove_variables_from_data_groups,
)

__all__ = [
    "CasaImageBackendEntrypoint",
    "FitsImageBackendEntrypoint",
    "open_image_dataset",
]

#: Read block of the backends: one plane per block (the time axis of CASA and
#: FITS images always has length one).
PLANE_CHUNKS = {"frequency": 1, "polarization": 1}


def open_image_dataset(
    path: str,
    *,
    drop_variables: str | Iterable[str] | None = None,
    image_type: str | None = None,
    image_chunks: Mapping[str, int] | None = None,
    do_sky_coords: bool = True,
    verbose: bool = False,
    compute_mask: bool = True,
) -> xr.Dataset:
    """Open a CASA or FITS image as a lazily read xradio image dataset.

    This is the implementation of the ``xradio_casa_image`` and
    ``xradio_fits_image`` xarray engines; the keyword arguments below can be
    passed to ``xr.open_dataset`` with those engines.

    Parameters
    ----------
    path : str
        Path to the CASA image directory or FITS file.
    drop_variables : str or iterable of str, optional
        Variable (or variables) to leave out of the dataset. A string is one
        name. Data group roles that refer to a dropped variable are removed
        from the ``data_groups`` attribute, and so are the ``flag`` and
        ``beam_fit_params`` attributes of the images that name it.
    image_type : str, optional
        Role of the image, as the keys of the dict form of
        :func:`xradio.image.open_image`: it gives the data variable and its
        data group role, e.g. ``"sky"`` (``SKY``), ``"point_spread_function"``
        or ``"psf"`` (``POINT_SPREAD_FUNCTION``), ``"primary_beam"``,
        ``"residual"`` (``SKY_RESIDUAL``), ``"mask"`` or ``"aperture"``. By
        default the role is taken from the image as
        :func:`xradio.image.open_image` does for a path (from the name, e.g.
        ``*.psf`` is a point spread function).
    image_chunks : dict, optional
        Size of the blocks in which the image is read, by dimension
        (``"frequency"``, ``"polarization"``, ``"l"`` / ``"u"``, ``"m"`` /
        ``"v"``); missing dimensions are read whole except ``frequency`` and
        ``polarization``, which default to 1 (one plane per block). The block
        sizes are the variables' ``encoding["preferred_chunks"]``. Each read
        of part of a block reads the whole block, so use blocks that match
        the dask ``chunks`` passed to ``xr.open_dataset``.
    do_sky_coords : bool, default True
        Compute the sky coordinates (``right_ascension`` and ``declination``
        or their galactic equivalents) of every pixel.
    verbose : bool, default False
        Emit debugging messages.
    compute_mask : bool, default True
        FITS only: flag the NaN pixels in a ``FLAG_*`` variable. This scans
        the image once when it is opened.

    Returns
    -------
    xarray.Dataset
        The image dataset, with data variables that read the image on access.
    """
    from xradio.image import open_image

    if image_type is None:
        store = path
    elif isinstance(image_type, str):
        _check_image_type(image_type)
        store = {image_type: path}
    else:
        raise TypeError(
            f"image_type must be a str such as 'sky', not {type(image_type).__name__}"
        )

    xds = open_image(
        store,
        chunks={**PLANE_CHUNKS, **dict(image_chunks or {})},
        verbose=verbose,
        do_sky_coords=do_sky_coords,
        compute_mask=compute_mask,
    )
    if drop_variables is not None:
        xds = _drop_variables(xds, drop_variables)
    return _read_lazily(xds)


def _check_image_type(image_type: str) -> None:
    """Raise for an ``image_type`` that names no image role.

    Accepted are the image types of the data group roles (``sky``,
    ``point_spread_function``, ``primary_beam``, ``mask``, ...) and their
    tclean names (``psf``, ``pb``, ``sumwt``, ``residual``, ...), in any case,
    and the sky images of other data groups, ``sky_<group>``.
    """
    from xradio.image._util.image_factory import (
        _IMAGE_TYPE_ALIASES,
        _IMAGE_TYPE_OF_NAME_TOKEN,
        _normalize_image_type,
    )

    known = {
        value for value in _IMAGE_TYPE_OF_NAME_TOKEN.values() if value != "ALL"
    } | set(_IMAGE_TYPE_ALIASES.values())
    normalized = _normalize_image_type(image_type)
    if normalized in known or (
        normalized.startswith("SKY_") and len(normalized) > len("SKY_")
    ):
        return
    examples = sorted(
        name.lower()
        for name in known
        if not name.startswith("SKY_") or name in ("SKY_MODEL", "SKY_RESIDUAL")
    )
    raise ValueError(
        f"image_type {image_type!r} is not an image role; use one of "
        f"{', '.join(examples)}, a tclean product name such as 'psf' or 'pb', "
        "or 'sky_<group>' for the sky image of another data group"
    )


def _drop_variables(xds: xr.Dataset, drop_variables) -> xr.Dataset:
    """Drop variables as xarray's own backends do (a str is one name, names
    not in the dataset are ignored) and remove the data group roles and
    variable attributes (e.g. ``SKY.attrs["flag"]``) that refer to the
    dropped variables."""
    if isinstance(drop_variables, str):
        drop_variables = [drop_variables]
    names = [name for name in drop_variables if name in xds.variables]
    if not names:
        return xds
    xds = xds.drop_vars(names)
    remove_variables_from_data_groups(xds, names)
    remove_variable_references(xds, names)
    return xds


def _read_lazily(xds: xr.Dataset) -> xr.Dataset:
    """Replace the dask arrays of the data variables by lazily indexed arrays
    that read the image only when they are indexed or loaded."""
    import dask.array as da

    variables = {}
    for name, variable in xds.data_vars.items():
        data = variable.variable._data
        if not isinstance(data, da.Array):
            continue
        if "beam_params_label" in variable.dims:
            # Beam fit parameters come from the image metadata, which the
            # readers hold in memory already: load them, so that they do not
            # advertise one read block along the plane dimensions
            variables[name] = variable.variable.copy(
                data=data.compute(scheduler="synchronous")
            )
            continue
        array = _DaskBlockArray(data)
        variables[name] = xr.Variable(
            variable.dims,
            indexing.LazilyIndexedArray(array),
            attrs=variable.attrs,
            encoding={
                **variable.encoding,
                "preferred_chunks": _preferred_chunks(variable.dims, data.chunks),
            },
        )
    return xds.assign(variables) if variables else xds


def _preferred_chunks(dims: tuple, chunks: tuple) -> dict:
    """Dask chunks as an ``encoding["preferred_chunks"]`` mapping: a size per
    dimension when the chunks are regular, else the tuple of chunk sizes."""
    preferred = {}
    for dim, sizes in zip(dims, chunks, strict=True):
        if all(size == sizes[0] for size in sizes[:-1]) and sizes[-1] <= sizes[0]:
            preferred[dim] = int(sizes[0])
        else:
            preferred[dim] = tuple(int(size) for size in sizes)
    return preferred


class _DaskBlockArray(BackendArray):
    """A dask array read one block at a time.

    Indexing computes only the blocks that hold the requested elements, each
    from its own small task graph: slicing or computing the whole dask array
    instead would process its complete graph (one task per image plane) on
    every access, which costs O(planes) per read.
    """

    def __init__(self, array):
        from dask.core import flatten

        self.shape = tuple(array.shape)
        self.dtype = np.dtype(array.dtype)
        # Block start offsets per axis, with the axis length appended
        self._offsets = tuple(
            np.concatenate([[0], np.cumsum(sizes)]) for sizes in array.chunks
        )
        self._graph = dict(array.__dask_graph__())
        keys = np.empty(array.numblocks, dtype=object)
        for key in flatten(array.__dask_keys__()):
            keys[key[1:]] = key
        self._keys = keys

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem
        )

    def _block(self, block_index: tuple) -> np.ndarray:
        """Compute one block from the tasks it depends on only."""
        from dask.local import get_sync
        from dask.optimization import cull

        key = self._keys[block_index]
        graph, _ = cull(self._graph, [key])
        # Synchronous: this runs inside a dask task when the dataset is
        # chunked, and the block graph is a few tasks reading one block
        return np.asarray(get_sync(graph, key))

    def _getitem(self, key: tuple) -> np.ndarray:
        # Per axis: the selected indices grouped by block, as
        # (block number, selection in the block, selection in the output)
        axis_groups = []
        out_shape = []
        for k, size, offsets in zip(key, self.shape, self._offsets, strict=True):
            if isinstance(k, slice):
                indices = np.arange(*k.indices(size))
            else:
                indices = np.atleast_1d(np.asarray(k, dtype=np.intp))
                indices = np.where(indices < 0, indices + size, indices)
            if not isinstance(k, slice | np.ndarray):
                out_shape.append(None)  # integer index: no output axis
            else:
                out_shape.append(len(indices))
            axis_groups.append(_group_by_block(indices, offsets))

        out = np.empty([1 if n is None else n for n in out_shape], dtype=self.dtype)
        if out.size:
            for groups in itertools.product(*axis_groups):
                block = self._block(tuple(group[0] for group in groups))
                selected = _outer_select(block, [group[1] for group in groups])
                out[tuple(group[2] for group in groups)] = selected
        return out.reshape([n for n in out_shape if n is not None])


def _group_by_block(indices: np.ndarray, offsets: np.ndarray) -> list[tuple]:
    """Group sorted indices along one axis by the block that holds them.

    Returns a list of (block number, selection in the block, selection in the
    output) tuples; a selection is a slice when it can be one.
    """
    if not len(indices):
        return []
    blocks = np.searchsorted(offsets, indices, side="right") - 1
    breaks = np.flatnonzero(np.diff(blocks)) + 1
    groups = []
    for start, stop in zip(
        np.concatenate([[0], breaks]),
        np.concatenate([breaks, [len(indices)]]),
        strict=True,
    ):
        block = int(blocks[start])
        local = indices[start:stop] - offsets[block]
        groups.append((block, _as_slice(local), slice(int(start), int(stop))))
    return groups


def _as_slice(local: np.ndarray) -> slice | np.ndarray:
    """An index array as an equivalent slice when it is an increasing
    arithmetic progression, else unchanged."""
    if len(local) == 1:
        return slice(int(local[0]), int(local[0]) + 1)
    step = int(local[1] - local[0])
    if step > 0 and np.all(np.diff(local) == step):
        return slice(int(local[0]), int(local[-1]) + 1, step)
    return local


def _outer_select(block: np.ndarray, selections: list) -> np.ndarray:
    """Index a block with one slice or index array per axis, independently
    per axis (outer indexing)."""
    block = block[tuple(s if isinstance(s, slice) else slice(None) for s in selections)]
    for axis, selection in enumerate(selections):
        if not isinstance(selection, slice):
            block = np.take(block, selection, axis=axis)
    return block
