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

ASDM Xarray engine (`xradio_asdm`)
----------------------------------

Xradio includes an `Xarray backend <https://docs.xarray.dev/en/latest/api/backends.html>`__ for opening ASDMs (ALMA Science Data Model) as :py:class:`xarray.DataTree` s.
When Xradio is installed this backend can be given in the `engine` parameter of `xarray.open_datatree() <https://docs.xarray.dev/en/latest/generated/xarray.open_datatree.html>`__,
as `engine="xradio_asdm"`. The backend is implemented via a custom Xarray `BackendEntrypoint <https://docs.xarray.dev/en/latest/generated/xarray.backends.BackendEntrypoint.html>`__.
The parameters supported are as listed in the following `open_asdm()` function which is used by the backend to open ASDMs:

   .. autofunction:: xradio.measurement_set._utils._asdm.open_asdm.open_asdm
