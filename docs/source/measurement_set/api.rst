API documentation
=================

.. automodule:: xradio.measurement_set

.. autofunction:: open_processing_set

.. autofunction:: load_processing_set

.. autofunction:: convert_msv2_to_processing_set

.. autofunction:: estimate_conversion_memory_and_cores

MSv2 Xarray engine (``xradio_msv2``)
------------------------------------

Xradio includes an `Xarray backend <https://docs.xarray.dev/en/latest/api/backends.html>`__ that opens a CASA
MeasurementSet v2 directly as a Processing Set, without converting it: the :py:class:`xarray.DataTree` of MSv4s that
:py:func:`convert_msv2_to_processing_set` would write, as :py:func:`open_processing_set` would open it (the same MSv4
names, variables, values and attributes, and with ``chunks={}`` the same Dask chunks of the main data variables). It
reads the MS with python-casacore (``pip install "xradio[casacore]"``) or, where python-casacore is not installed (for
example on macOS), with casatools.

.. code:: python

   import xarray as xr

   ps_xdt = xr.open_datatree("my.ms", engine="xradio_msv2", chunks={})

   # or, with the conveniences of open_processing_set (array_backend, scan_intents):
   from xradio.measurement_set import open_msv2

   ps_xdt = open_msv2("my.ms", partition_scheme=["FIELD_ID"])

The engine accepts the options of the converter that shape the processing set (``partition_scheme``,
``partition_filter``, ``main_chunksize``, ``with_pointing``, ``pointing_chunksize`` and the ``*_interpolate`` options),
and ``drop_variables``, ``partition_cache`` and ``on_partition_error`` (see :py:func:`open_msv2`). A partition that
cannot be opened is left out with a ``RuntimeWarning`` (``on_partition_error="skip"``, the default; the other MSv4s
keep their names); ``on_partition_error="raise"`` raises instead. ``xr.open_dataset`` cannot open an MS: use
``xr.open_datatree``.

**What is read when.** Opening reads the metadata of the MS: its partitions (stored in the MS, see below), and for
every partition the coordinates and the sub-datasets (antenna, field and source, system calibration, ...) as the
converter builds them. The main data variables (``VISIBILITY`` or ``SPECTRUM`` and those of the other data groups,
``FLAG``, ``WEIGHT``, ``UVW``, ``TIME_CENTROID``, ``EFFECTIVE_INTEGRATION_TIME``) are read from the MAIN table only
when they are indexed or computed, block by block (whole cells, at most 128 MiB at a time). Cells without a MAIN row
are NaN, and their ``FLAG`` is False, as in the converted processing set. The ``pointing_xds`` is lazy too: opening
reads only the POINTING ``TIME`` and ``ANTENNA_ID`` columns. It is read when the MS is opened, as by the converter,
with ``pointing_interpolate=True`` and for POINTING tables that cannot be read in blocks (cells of several shapes or
empty cells, unusual value types, more than about 16.7 million rows).

**Chunks.** Open with ``chunks={}``: every variable is a Dask array, and the main data variables have the chunks of the
converter (``main_chunksize``, by default about 128 MiB along time). ``chunks="auto"`` lets Dask choose other chunks
for every variable. Without ``chunks``, the variables are lazily indexed arrays: a selection reads only its rows, but a
variable loaded whole stays in memory as long as the tree; pass ``cache=False`` to scan through the data. The
variables of a ``pointing_xds`` are one chunk each, as in the converted processing set; for large POINTING tables,
give ``pointing_chunksize`` (for example ``{"time": 10000}``).

**Partitions stored in the MS: opening may write.** To compute the partitions, the key columns of the whole MAIN table
are read, which takes time for large MSs. So the first open of a writable MS stores them in the MS:

- a sub-table ``XRADIO_PARTITIONS``;
- a MAIN keyword of the same name that links it (casacore rewrites MAIN's ``table.dat``, and its ``table.info`` in
  place);
- a row of the ``HISTORY`` table.

Later opens read the stored partitions (in milliseconds) while the MS is unchanged, and every process also keeps them
in memory. ``partition_cache`` sets what is done:

- ``"auto"`` (default): use the stored partitions, or compute and store them;
- ``"read"``: use the stored partitions, never write (for archives and shared data);
- ``"rebuild"``: compute the partitions and store them (a ``HISTORY`` row only if they changed);
- ``"off"``: compute the partitions, neither use nor store them.

The default is the value of the environment variable ``XRADIO_MSV2_PARTITION_CACHE`` if it is set, else ``"auto"``. A
read-only MS is opened with partitions computed in memory, and a :py:class:`PartitionCacheWarning` (once per MS and
reason) says so; so is an MS whose MAIN table another process has locked (for example a CASA session with the MS open):
xradio never waits for a lock. With casatools the partitions are never stored (logged once). If the process that opens
the MS also holds it open with python-casacore's default (automatic) locking, storing switches that table to user
locking (once per MS): pass ``partition_cache="read"`` in such sessions. CASA tasks that copy an MS (``split``,
``mstransform``, ``tb.copy``, ``msconcat``) also copy the sub-table: the copy's first open finds that it does not apply
and computes the partitions again. :py:func:`remove_msv2_partition_cache` removes the stored partitions.

**When the MS changes.** The stored partitions are used only while a fingerprint of the tables they are computed from
(the data managers and key columns of MAIN, and the FIELD, STATE and SOURCE tables, read from casacore's lock files) is
unchanged and the ``HISTORY`` table has no rows newer than xradio's own. They are also checked against their MAIN rows
when the MS is opened. An MS that changes after it was opened is not followed: a lazy read then raises
:py:class:`MSv2ChangedError` (MAIN or POINTING has other rows), and the MS must be opened again.

**Performance.** Opening costs what the converter spends on metadata: about 0.05 to 0.2 s per partition, plus 4 to
12 ms per node for xarray (about 2.6 s and 150 MiB for the 20 partitions of a 160 MB VLASS MS). Opening more than
1,000 partitions gives a warning. For large MSs, keep the default ``partition_scheme=[]``, select partitions with
``partition_filter``, and pass ``with_pointing=False`` when the pointing is not needed. python-casacore holds the
Python GIL while it reads, so reading with threads is not faster: use processes (Dask's ``processes`` scheduler, or a
``distributed.LocalCluster(processes=True)``); with casatools, reads are serialized by one process-wide lock.
Converting is still the better choice when the data is read in full several times, or from cloud storage: every read
from the MS reorders its rows into the MSv4 layout, which the converted processing set stores once, chunked and
compressed.

With `xarray-ms <https://github.com/ratt-ru/xarray-ms>`__ also installed, pass ``engine="xradio_msv2"`` (or use
:py:func:`open_msv2`): without it, xarray tries the engines in the order of their names, and xarray-ms's engine claims
every CASA table.

.. autofunction:: open_msv2

.. autofunction:: remove_msv2_partition_cache

.. autoexception:: MSv2ChangedError

.. autoexception:: MSv2ReadError

.. autoexception:: PartitionCacheWarning

ProcessingSetXdt API
--------------------

Custom accessor to Processing Set additional functionality. Given a Processing Set :py:class:`xarray.DataTree`, named `ps_xdt`,
the accessor can be used as `ps_xdt.xr_ps` (`xr` for xradio and `ps` for Processing Set).

   .. autoclass:: ProcessingSetXdt
      :members:


MeasurementSetXdt API
---------------------

Custom accessor to MSv4 additional functionality. Given an MSv4 :py:class:`xarray.DataTree`, named `ms_xdt`,
the accessor can be used as `ms_xdt.xr_ms` (`xr` for xradio and `ms` for Measurement Set).

   .. autoclass:: MeasurementSetXdt
      :members:
