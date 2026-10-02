API documentation
=================

.. automodule:: xradio.measurement_set

.. autofunction:: open_processing_set

.. autofunction:: load_processing_set

.. autofunction:: convert_msv2_to_processing_set

.. autofunction:: estimate_conversion_memory_and_cores

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

ASDM Xarray engine (``xradio_asdm``)
------------------------------------

Xradio includes an `Xarray backend <https://docs.xarray.dev/en/latest/api/backends.html>`__ for opening ASDMs (ALMA Science Data Model)
as Processing Sets (:py:class:`xarray.DataTree` s of MSv4s). The backend needs the optional ASDM dependencies, in particular
`pyasdm <https://github.com/casangi/pyasdm>`__, which are installed with the ``asdm`` extra (they are also included in
``xradio[all]``):

.. code:: sh

   pip install "xradio[asdm]"

With these dependencies installed, the backend can be given in the ``engine`` parameter of
`xarray.open_datatree() <https://docs.xarray.dev/en/latest/generated/xarray.open_datatree.html>`__, as ``engine="xradio_asdm"``:

.. code:: python

   import xarray as xr

   ps_xdt = xr.open_datatree("uid___A002_X10f6291_X12d9", engine="xradio_asdm")

The ASDM is opened lazily: opening reads the ASDM metadata tables, and the correlated data (``VISIBILITY`` or
``SPECTRUM``) and ``FLAG`` are read from the ASDM binary data files (BDFs) only when they are accessed or loaded.
``UVW`` is computed from the metadata (antenna positions, phase center and times) when it is accessed. ``WEIGHT`` is
currently a constant 1.0, as the ASDM binary data have no weights. Without ``chunks``, loading or saving (for example
with ``to_zarr()``) reads each data variable of each MSv4 whole into memory. For large ASDMs, open with ``chunks`` to
get Dask arrays that are read chunk by chunk. With ``chunks={}``, every MSv4 gets time chunks of one size (the last
chunk can be smaller), shared by all its data variables: about the number of integrations of its largest BDF, fewer when
needed to fit the Dask chunk size (``array.chunk-size``). The chunks line up with the BDFs only when all the BDFs of the
MSv4 have the same number of integrations (the last one can have fewer). Otherwise some chunks span two BDFs, which
makes reading them slower. With ``chunks="auto"`` Dask can also merge several of these chunks into one, up to its chunk
size (decided separately for every data variable), which gives fewer and larger chunks when the BDFs are small. Explicit
chunk sizes (for example ``chunks={"time": 10}``) can also be given. Chunks that split BDFs make reading slower, and
xarray warns when explicit chunks "separate the stored chunks", that is the chunks that ``chunks={}`` would give.

The backend is implemented via a custom Xarray `BackendEntrypoint <https://docs.xarray.dev/en/latest/generated/xarray.backends.BackendEntrypoint.html>`__.
The parameters it supports are those of the following ``open_asdm()`` function, which is used by the backend to open
ASDMs. It can also be called directly as ``xradio.measurement_set.open_asdm()`` (with the same optional dependencies
installed). See also the :doc:`ALMA ASDM guide <guides/ALMA_ASDM_IF>` and the
:doc:`ASDM backend performance notes <../performance/asdm/asdm_backend_performance>`.

.. autofunction:: open_asdm
