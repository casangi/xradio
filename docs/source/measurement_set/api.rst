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
and ``drop_variables``, ``skip_columns``, ``partition_cache`` and ``on_partition_error`` (see :py:func:`open_msv2`). A
partition that cannot be opened is left out with a ``RuntimeWarning`` (``on_partition_error="skip"``, the default; the
other MSv4s keep their names); ``on_partition_error="raise"`` raises instead. An MS without MAIN rows opens as an empty
processing set, as the converter writes it. ``xr.open_dataset`` cannot open an MS: use ``xr.open_datatree``.

**What is read when.** Opening reads the metadata of the MS: its partitions (stored in the MS, see below), and for every
partition the coordinates and the sub-datasets (antenna, field and source, system calibration, ...) as the converter
builds them. The main data variables (``VISIBILITY`` or ``SPECTRUM`` and those of the other data groups, ``FLAG``,
``WEIGHT``, ``UVW``, ``TIME_CENTROID``, ``EFFECTIVE_INTEGRATION_TIME``) are read from the MAIN table only when they are
indexed or computed: only the rows of the selected times and baselines, in whole cells, at most 128 MiB at a time. With
python-casacore, a selection of some channels of ``VISIBILITY*``, ``SPECTRUM``, ``FLAG`` or ``WEIGHT`` (from
``WEIGHT_SPECTRUM``) whose MAIN column is stored in tiles of fewer channels than a cell (tiled storage managers) reads
only the tiles of those channels: their range rounded out to whole tiles. Cells without a MAIN row are NaN, and their
``FLAG`` is False, as in the converted processing set. The ``pointing_xds`` is lazy too, whatever the size of the
POINTING table: opening reads its ``TIME`` and ``ANTENNA_ID`` columns, the shapes
(not the values) of the cells of its array columns (stored in the MS by the first open, see below) and one cell of each
data column, and a selection reads only the rows of its times and antennas. Only with ``pointing_interpolate=True`` is
it read when the MS is opened, as by the converter. The converter reads the POINTING rows of every partition on their
own, and leaves out (1,000 rows or more) or pads (fewer) a column whose cells there have several shapes or no value: a
partition whose POINTING rows have such cells gets the converter's ``pointing_xds`` from the shapes of its cells (no
value is read at open), with variables read lazily as above (1,000 rows or more) or built by the converter's code when
they are read (fewer, once for all of them). Every partition of a POINTING table that cannot be described without
reading it (no ``DIRECTION`` column, unusual value types) has its ``pointing_xds`` built by the converter's code when
the MS is opened (the values are not kept) and again when its variables are read.

**Columns only a read can check.** The converter leaves out a MAIN column whose cells cannot be read (for
``WEIGHT_SPECTRUM`` it reads ``WEIGHT`` instead). For most storage managers the engine finds such cells when the MS is
opened and does the same. Where only a read can tell (cells of several shapes in a ``StandardStMan`` column, reference
and concatenated MSs), the variable is opened and its read raises :py:class:`MSv2ReadError`, which names the remedies:
``skip_columns=["<column>"]`` gives the converter's processing set (the column treated as unreadable in every
partition), ``drop_variables=["<variable>"]`` leaves out only that variable. Until then the opened processing set
differs from the converted one where the converter dropped the column: it has the variable (for a data column also its
data group and ``field_and_source_xds``), and for ``WEIGHT_SPECTRUM`` its ``WEIGHT`` has the attributes of that column
rather than those of ``WEIGHT``. (The cells of POINTING columns are checked when the MS is opened, see above.)

**Chunks.** Open with ``chunks={}``: every variable is a Dask array, and the main data variables have the chunks of the
converter (``main_chunksize``, by default about 128 MiB along time). The other variables (coordinates, and the
variables of the sub-datasets such as ``system_calibration_xds``) are one chunk each, but for the time coordinates of
the main dataset (such as ``scan_name`` and ``field_name``), which have the time chunks of the data so that
``Dataset.chunks`` is defined; the converted processing set has the chunks zarr chose for these variables when it was
written. The values are the same, but ``to_zarr`` of an opened tree may chunk these variables differently from the
converter. ``chunks="auto"`` lets Dask choose other chunks for every variable. Without ``chunks``, the variables are
lazily indexed arrays: a selection reads only its rows, but a variable loaded whole stays in memory as long as the tree;
pass ``cache=False`` to scan through the data. The variables of a ``pointing_xds`` are one chunk each, as in the
converted processing set; for large POINTING tables, give ``pointing_chunksize`` (for example ``{"time": 10000}``).

**Partitions stored in the MS: opening may write.** To compute the partitions, the key columns of the whole MAIN table
are read, which takes time for large MSs. So the first open of a writable MS stores them in the MS:

- a sub-table ``XRADIO_PARTITIONS``;
- a MAIN keyword of the same name that links it (casacore rewrites MAIN's ``table.dat``, and its ``table.info`` in
  place);
- a row of the ``HISTORY`` table.

Later opens read the stored partitions (in milliseconds) while the MS is unchanged, and every process also keeps them
in memory. The sub-table also stores the shapes of the cells of the POINTING array columns, which the first open scans
(about 0.2 s per column for 550,000 rows), with a fingerprint of the POINTING table: later opens, in any process, use
them while that table is unchanged (no ``HISTORY`` row is written for them). ``partition_cache`` sets what is done:

- ``"auto"`` (default): use the stored partitions, or compute and store them;
- ``"read"``: use the stored partitions, never write (for archives and shared data);
- ``"rebuild"``: compute the partitions and store them (a ``HISTORY`` row only if they changed);
- ``"off"``: compute the partitions, neither use nor store them.

The default is the value of the environment variable ``XRADIO_MSV2_PARTITION_CACHE`` if it is set, else ``"auto"``. A
read-only MS is opened with partitions computed in memory, and a :py:class:`PartitionCacheWarning` (once per MS and
reason) says so; so is an MS whose MAIN table another process has locked (for example a CASA session with the MS open):
xradio never waits for a lock. With casatools the partitions are never stored (logged once). Those of reference and
concatenated MSs (a multi-MS), and of MSs whose MAIN key columns are forwarded to other MSs (as ``msconcat`` makes them),
are computed on every open, neither stored nor kept in memory (logged once): their rows live in other tables, whose
changes their lock files do not show. CASA tasks that copy an MS (``split``, ``mstransform``, ``tb.copy``, ``msconcat``)
also copy the sub-table: the copy's first open finds that it does not apply and computes the partitions again.
:py:func:`remove_msv2_partition_cache` removes the stored partitions (only a sub-table that xradio wrote, and not while
another process holds a lock on it); an open in another process at the same time stores a sub-table of its own.

**A writable table handle in the same process.** Within one process, python-casacore shares one table object per
table, with its locks, and closing any handle of a table releases them. Every open of an MS by xradio (the engine in
every ``partition_cache`` mode, as the converter) opens and closes the tables of the MS, so it releases the write lock
of a writable MAIN handle that the process holds, which flushes the handle's changes first (the open sees them), unless
that handle was opened with ``lockoptions="permanent"``. A handle opened with python-casacore's default lock options is
also switched to user locking: its next write raises ``... should be locked when using UserLocking`` unless it calls
``lock()`` first (as a handle opened with ``lockoptions="user"`` must anyway). So open such handles with
``lockoptions="auto"`` (their writes take the lock again) or ``"permanent"``, or close them before the MS is opened by
xradio. While the process holds the MAIN write lock (``"permanent"``), the partitions are not stored; if the handle
added rows that it has not flushed, the partitions are computed from the rows the process sees, as the converter reads
them, without the stored ones. Likewise, a thread that closes a MAIN handle while another thread stores the partitions
releases the MAIN write lock: the store takes the lock again before it writes and flushes the MAIN keyword; between
these steps another process could take the lock and write MAIN too (a window of a few Python statements, once per MS).

**When the MS changes.** The stored partitions are used only while a fingerprint of the tables they are computed from
(the data managers and key columns of MAIN, and the FIELD, STATE and SOURCE tables, read from casacore's lock files) is
unchanged and the ``HISTORY`` table has no rows newer than xradio's own. They are also checked against their MAIN rows
when the MS is opened, and so are partitions computed while the MS may have changed (if they do not describe their rows
because the MS changed meanwhile, the open is done again with the partitions computed). An MS that changes after it was
opened is not followed: a lazy read returns the current values of the rows of its partition, or raises
:py:class:`MSv2ChangedError`, and the MS must then be opened again. It raises if MAIN or POINTING has another number of
rows, if the partition no longer has exactly the rows it had when the MS was opened (rows moved to or from another
partition because their ``DATA_DESC_ID``, ``OBSERVATION_ID``, observing mode, ephemeris or a key of the
``partition_scheme`` changed, in MAIN or in the ``FIELD``, ``STATE`` or ``SOURCE`` rows they refer to), or if the (time,
baseline) grid of the partition, or the times, antennas and rows of its ``pointing_xds``, changed. To tell, a read first
looks at casacore's lock files: while the key columns of MAIN (and their data managers) and the FIELD, STATE and SOURCE
tables have not been written since the open, nothing more is read. Otherwise the whole partition is checked against the
MS as it is now (the key columns of every MAIN row are read), once per state of the MS in a process, and the rows a read
reads are checked against the times and antennas of their cells (MAIN ``TIME``, ``ANTENNA1``, ``ANTENNA2``; POINTING
``TIME``, ``ANTENNA_ID``), so the outcome does not depend on what the process kept in memory: values rewritten in place
are read as they are now, as are rows moved within an unchanged grid. While this process has the MAIN table open for
writing (its writes are seen before they are flushed), and for reference and concatenated MSs (their lock files do not
show the writes of the MSs they read), every read checks its partition so.

**Performance.** Opening costs what the converter spends on metadata: about 0.05 to 0.2 s per partition, plus 4 to
12 ms per node for xarray (about 3 s and 150 MiB for the 20 partitions of a 160 MB VLASS MS). The first open of an MS
also scans the cell shapes of the POINTING array columns (about 1 s per column for 3.6 million rows), which it stores in
the MS for later opens (an MS that cannot be written: once per process). Opening more than 1,000 partitions gives a
warning. For large MSs, keep the default ``partition_scheme=[]``, select partitions with
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
