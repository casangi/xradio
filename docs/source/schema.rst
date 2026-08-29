
Schema Support
==============

Data model schemas not only allow us to generate documentation,
but also check automatically whether :py:class:`xarray.DataArray` and
:py:class:`xarray.Dataset` objects conform to the :py:mod:`xradio` schemas (see
e.g. :py:class:`xradio.measurement_set.schema.VisibilityXds`).

Checking
--------

.. automodule:: xradio.schema.check
  :members:

Shared Measures
---------------

.. automodule:: xradio.schema.measures

.. autoclass:: xradio.schema.measures.TimeArray()

   .. xradio_array_schema_table:: xradio.schema.measures.TimeArray

.. autoclass:: xradio.schema.measures.SkyCoordArray()

   .. xradio_array_schema_table:: xradio.schema.measures.SkyCoordArray

.. autoclass:: xradio.schema.measures.SpectralCoordArray()

   .. xradio_array_schema_table:: xradio.schema.measures.SpectralCoordArray

.. autoclass:: xradio.schema.measures.LocationArray()

   .. xradio_array_schema_table:: xradio.schema.measures.LocationArray

.. autoclass:: xradio.schema.measures.DopplerArray()

   .. xradio_array_schema_table:: xradio.schema.measures.DopplerArray

.. autoclass:: xradio.schema.measures.PolarizationArray()

   .. xradio_array_schema_table:: xradio.schema.measures.PolarizationArray

.. autoclass:: xradio.schema.measures.QuantityInSecondsArray()

   .. xradio_array_schema_table:: xradio.schema.measures.QuantityInSecondsArray

.. autoclass:: xradio.schema.measures.QuantityInHertzArray()

   .. xradio_array_schema_table:: xradio.schema.measures.QuantityInHertzArray

.. autoclass:: xradio.schema.measures.QuantityInMetersArray()

   .. xradio_array_schema_table:: xradio.schema.measures.QuantityInMetersArray

.. autoclass:: xradio.schema.measures.QuantityInMetersPerSecondArray()

   .. xradio_array_schema_table:: xradio.schema.measures.QuantityInMetersPerSecondArray

.. autoclass:: xradio.schema.measures.QuantityInRadiansArray()

   .. xradio_array_schema_table:: xradio.schema.measures.QuantityInRadiansArray

Decorators
----------

.. automodule:: xradio.schema.bases
  :members:

Annotations
-----------

.. automodule:: xradio.schema.typing
  :members:

Data Model
----------

.. automodule:: xradio.schema.metamodel
  :members:

Import and Export
-----------------

.. automodule:: xradio.schema.export
  :members:
