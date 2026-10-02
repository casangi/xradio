"""
Read image arrays region by region for the CASA and FITS writers.

The writers read an image (and its flags or masks) in regions made of whole
dask chunks and write each region as soon as it has been computed, so that
the cube is never held in memory as a whole. Computing each region with
``dask.compute`` would cull and order the complete task graph on every call,
which makes a write quadratic in the number of chunks (the readers build a
graph with one layer per chunk). :class:`RegionReader` materializes the task
graph once and computes batches of regions from the tasks they depend on
only, with the active dask scheduler (threads, processes or a distributed
client), so that chunks are computed in parallel and the work per batch is
proportional to the batch.

This module only needs dask and numpy.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np
import xarray as xr

#: Largest amount of pixel data (bytes, summed over the arrays read) that is
#: computed in one batch of regions; a batch holds at least one region.
#: Batches let the dask scheduler compute several chunks in parallel while
#: the memory held at once stays bounded.
BATCH_BYTES = 256 * 2**20


class RegionReader:
    """Compute regions of several arrays of the same shape, batch by batch.

    Parameters
    ----------
    arrays : sequence of xr.DataArray or None
        The arrays to read, all of the same shape (None entries are skipped
        and yield None). Dask backed arrays are computed from their task
        graph; other arrays (numpy, or lazily indexed backend arrays) are
        indexed region by region.
    """

    def __init__(self, arrays: Sequence[xr.DataArray | None]):
        from dask.core import flatten

        self._arrays = list(arrays)
        self._dask = [
            array is not None and array.chunks is not None for array in self._arrays
        ]
        dask_arrays = [
            array.data
            for array, lazy in zip(self._arrays, self._dask, strict=True)
            if lazy
        ]
        # One materialized graph for all arrays (arrays derived from the same
        # reads, such as an image and the flags of its NaN pixels, share
        # tasks, which a batch then computes once)
        self._graph = {}
        for array in dask_arrays:
            self._graph.update(array.__dask_graph__())
        # The scheduler is looked up when regions are computed (see regions),
        # like dask.compute, so the caller's scheduler at that time applies
        self._dask_arrays = dask_arrays
        # Per dask array: the block start offsets along each axis (with the
        # axis length appended) and the task key of every block
        self._offsets = []
        self._keys = []
        for array, lazy in zip(self._arrays, self._dask, strict=True):
            if not lazy:
                self._offsets.append(None)
                self._keys.append(None)
                continue
            data = array.data
            self._offsets.append(
                [np.concatenate([[0], np.cumsum(sizes)]) for sizes in data.chunks]
            )
            keys = np.empty(data.numblocks, dtype=object)
            for key in flatten(data.__dask_keys__()):
                keys[key[1:]] = key
            self._keys.append(keys)

    def _blocks(self, index: int, region: tuple) -> list[tuple[tuple, tuple]]:
        """Blocks of dask array ``index`` that overlap ``region``.

        Returns
        -------
        list of (tuple of int, tuple of slice)
            For each block its block index and its slices in the cube.
        """
        ranges = []
        for offsets, selection in zip(self._offsets[index], region, strict=True):
            first = int(np.searchsorted(offsets, selection.start, side="right")) - 1
            last = int(np.searchsorted(offsets, selection.stop, side="left"))
            ranges.append(range(first, last))
        blocks = []
        for block in np.ndindex(*(len(r) for r in ranges)):
            block_index = tuple(r[i] for r, i in zip(ranges, block, strict=True))
            slices = tuple(
                slice(int(offsets[i]), int(offsets[i + 1]))
                for offsets, i in zip(self._offsets[index], block_index, strict=True)
            )
            blocks.append((block_index, slices))
        return blocks

    def _assemble(self, index: int, region: tuple, results: dict) -> np.ndarray:
        """The values of ``region`` of dask array ``index``, from computed
        blocks (a view of the block when a single block covers the region)."""
        blocks = self._blocks(index, region)
        values = []
        for block_index, slices in blocks:
            values.append((slices, np.asarray(results[self._keys[index][block_index]])))
        if len(values) == 1:
            slices, block = values[0]
            return block[
                tuple(
                    slice(r.start - s.start, r.stop - s.start)
                    for r, s in zip(region, slices, strict=True)
                )
            ]
        out = np.empty(
            tuple(r.stop - r.start for r in region), dtype=values[0][1].dtype
        )
        for slices, block in values:
            overlap = [
                (max(r.start, s.start), min(r.stop, s.stop))
                for r, s in zip(region, slices, strict=True)
            ]
            out[
                tuple(
                    slice(a - r.start, b - r.start)
                    for (a, b), r in zip(overlap, region, strict=True)
                )
            ] = block[
                tuple(
                    slice(a - s.start, b - s.start)
                    for (a, b), s in zip(overlap, slices, strict=True)
                )
            ]
        return out

    def _region_bytes(self, region: tuple) -> int:
        size = int(np.prod([r.stop - r.start for r in region]))
        return sum(
            size * array.dtype.itemsize for array in self._arrays if array is not None
        )

    def _compute_batch(self, regions: list, schedule) -> Iterator[tuple[tuple, list]]:
        from dask.optimization import cull

        keys = []
        seen = set()
        for region in regions:
            for index, lazy in enumerate(self._dask):
                if not lazy:
                    continue
                for block_index, _ in self._blocks(index, region):
                    key = self._keys[index][block_index]
                    if key not in seen:
                        seen.add(key)
                        keys.append(key)
        results = {}
        if keys:
            # only the tasks the batch depends on, so that a batch costs in
            # proportion to its size, not to the size of the whole graph
            graph, _ = cull(self._graph, keys)
            results = dict(zip(keys, schedule(graph, keys), strict=True))
        for region in regions:
            values = []
            for index, (array, lazy) in enumerate(
                zip(self._arrays, self._dask, strict=True)
            ):
                if array is None:
                    values.append(None)
                elif lazy:
                    values.append(self._assemble(index, region, results))
                else:
                    values.append(np.asarray(array[region].values))
            yield region, values

    def regions(
        self, regions: Iterable[tuple], batch_bytes: int | None = None
    ) -> Iterator[tuple[tuple, list]]:
        """Compute regions of the arrays, in the order given.

        Parameters
        ----------
        regions : iterable of tuple of slice
            Regions of the arrays (one slice with start and stop per
            dimension). A chunk of a dask array is computed once per batch
            whose regions overlap it, so regions made of whole chunks of
            every array compute each chunk exactly once.
        batch_bytes : int, optional
            Largest amount of data computed in one batch (by default
            :data:`BATCH_BYTES`).

        Yields
        ------
        region : tuple of slice
            The region.
        values : list of np.ndarray or None
            The values of each array in the region (None for a None array).
        """
        from dask.base import get_scheduler

        if batch_bytes is None:
            batch_bytes = BATCH_BYTES
        # The scheduler active when the regions are computed (for example
        # inside dask.config.set(scheduler=...)), not when the reader was made
        schedule = (
            get_scheduler(collections=self._dask_arrays) if self._dask_arrays else None
        )
        batch = []
        size = 0
        for region in regions:
            region_bytes = self._region_bytes(region)
            if batch and size + region_bytes > batch_bytes:
                yield from self._compute_batch(batch, schedule)
                batch, size = [], 0
            batch.append(region)
            size += region_bytes
        if batch:
            yield from self._compute_batch(batch, schedule)
