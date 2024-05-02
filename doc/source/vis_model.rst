
Visibility data model
=====================

.. _visibility datasets:

Data sets
---------

Model of visibility :py:class:`xarray.Dataset`: A collection of
:ref:`Visibility arrays` and :ref:`visibility Coordinates` sharing the same
dimensions, forming a comprehensive view of visibility data.

.. autoclass:: xradio.vis.schema.VisibilityXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.VisibilityXds

.. autoclass:: xradio.vis.schema.AntennaXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.AntennaXds

.. autoclass:: xradio.vis.schema.PointingXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.PointingXds

.. _visibility arrays:

Data Arrays
-----------

Models of visibility :py:class:`xarray.DataArray` s.
Bulk data gathered into :ref:`Visibility Datasets`.

.. autoclass:: xradio.vis.schema.VisibilityArray()

   .. xradio_array_schema_table:: xradio.vis.schema.VisibilityArray

.. autoclass:: xradio.vis.schema.FlagArray()

   .. xradio_array_schema_table:: xradio.vis.schema.FlagArray

.. autoclass:: xradio.vis.schema.WeightArray()

   .. xradio_array_schema_table:: xradio.vis.schema.WeightArray

.. autoclass:: xradio.vis.schema.UvwArray()

   .. xradio_array_schema_table:: xradio.vis.schema.UvwArray

.. autoclass:: xradio.vis.schema.SkyCoordArray()

   .. xradio_array_schema_table:: xradio.vis.schema.SkyCoordArray

.. autoclass:: xradio.vis.schema.TimeSamplingArray()

   .. xradio_array_schema_table:: xradio.vis.schema.TimeSamplingArray

.. autoclass:: xradio.vis.schema.FreqSamplingArray()

   .. xradio_array_schema_table:: xradio.vis.schema.FreqSamplingArray

.. _visibility coordinates:

Coordinates
-----------

Define fundamental dimensions of Xradio data, and associate them with
values. The shape of all arrays contained in Xradio datasets will be defined by
a mapping of dimensions to sizes -- the "shape" of the array. For instance, a
visibility data array might have an associated channel or time step count. Use
:ref:`Visibility Coordinates` to associate dimension indicies with values such
as frequencies or timestamps.

.. autoclass:: xradio.vis.schema.TimeArray()

   .. xradio_array_schema_table:: xradio.vis.schema.TimeArray

.. autoclass:: xradio.vis.schema.BaselineArray()

   .. xradio_array_schema_table:: xradio.vis.schema.BaselineArray

.. autoclass:: xradio.vis.schema.BaselineAntennaArray()

   .. xradio_array_schema_table:: xradio.vis.schema.BaselineAntennaArray

.. autoclass:: xradio.vis.schema.FrequencyArray()

   .. xradio_array_schema_table:: xradio.vis.schema.FrequencyArray

.. autoclass:: xradio.vis.schema.PolarizationArray()

   .. xradio_array_schema_table:: xradio.vis.schema.PolarizationArray

.. autoclass:: xradio.vis.schema.UvwLabelArray()

   .. xradio_array_schema_table:: xradio.vis.schema.UvwLabelArray

.. autoclass:: xradio.vis.schema.QuantityArray()
                                                 
   .. xradio_array_schema_table:: xradio.vis.schema.QuantityArray

.. autoclass:: xradio.vis.schema.EarthLocationArray()
                                                 
   .. xradio_array_schema_table:: xradio.vis.schema.EarthLocationArray

.. _visibility attributes:

Attributes
----------

Attribute data

.. automodule:: xradio.vis.schema
   :members: SourceInfoDict, FieldInfoDict
   :undoc-members:
   :member-order: bysource
