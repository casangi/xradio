
Visibility data model
=====================


.. _visibility datasets:

Data sets
---------

Model of visibility :py:class:`xarray.Dataset`: A collection of
:ref:`Visibility arrays` and :ref:`visibility Coordinates` sharing the same
dimensions, forming a comprehensive view of visibility data.

.. autoclass:: xradio.vis.model.VisibilityXds()

   .. xradio_dataset_schema_table:: xradio.vis.model.VisibilityXds

.. autoclass:: xradio.vis.model.AntennaXds()

   .. xradio_dataset_schema_table:: xradio.vis.model.AntennaXds

.. autoclass:: xradio.vis.model.PointingXds()

   .. xradio_dataset_schema_table:: xradio.vis.model.PointingXds

.. _visibility arrays:

Data Arrays
-----------

Models of visibility :py:class:`xarray.DataArray` s.
Bulk data gathered into :ref:`Visibility Datasets`.

.. autoclass:: xradio.vis.model.VisibilityArray()

   .. xradio_array_schema_table:: xradio.vis.model.VisibilityArray

.. autoclass:: xradio.vis.model.FlagArray()

   .. xradio_array_schema_table:: xradio.vis.model.FlagArray

.. autoclass:: xradio.vis.model.WeightArray()

   .. xradio_array_schema_table:: xradio.vis.model.WeightArray

.. autoclass:: xradio.vis.model.UvwArray()

   .. xradio_array_schema_table:: xradio.vis.model.UvwArray

.. autoclass:: xradio.vis.model.SkyCoordArray()

   .. xradio_array_schema_table:: xradio.vis.model.SkyCoordArray

.. autoclass:: xradio.vis.model.TimeSamplingArray()

   .. xradio_array_schema_table:: xradio.vis.model.TimeSamplingArray

.. autoclass:: xradio.vis.model.FreqSamplingArray()

   .. xradio_array_schema_table:: xradio.vis.model.FreqSamplingArray

.. _visibility coordinates:

Coordinates
-----------

Define fundamental dimensions of Xradio data, and associate them with
values. The shape of all arrays contained in Xradio datasets will be defined by
a mapping of dimensions to sizes -- the "shape" of the array. For instance, a
visibility data array might have an associated channel or time step count. Use
:ref:`Visibility Coordinates` to associate dimension indicies with values such
as frequencies or timestamps.

.. autoclass:: xradio.vis.model.TimeArray()

   .. xradio_array_schema_table:: xradio.vis.model.TimeArray

.. autoclass:: xradio.vis.model.BaselineArray()

   .. xradio_array_schema_table:: xradio.vis.model.BaselineArray

.. autoclass:: xradio.vis.model.BaselineAntennaArray()

   .. xradio_array_schema_table:: xradio.vis.model.BaselineAntennaArray

.. autoclass:: xradio.vis.model.FrequencyArray()

   .. xradio_array_schema_table:: xradio.vis.model.FrequencyArray

.. autoclass:: xradio.vis.model.PolarizationArray()

   .. xradio_array_schema_table:: xradio.vis.model.PolarizationArray

.. autoclass:: xradio.vis.model.UvwLabelArray()

   .. xradio_array_schema_table:: xradio.vis.model.UvwLabelArray

.. autoclass:: xradio.vis.model.QuantityArray()
                                                 
   .. xradio_array_schema_table:: xradio.vis.model.QuantityArray

.. autoclass:: xradio.vis.model.EarthLocationArray()
                                                 
   .. xradio_array_schema_table:: xradio.vis.model.EarthLocationArray

.. _visibility attributes:

Attributes
----------

Attribute data

.. automodule:: xradio.vis.model
   :members: SourceInfoDict, FieldInfoDict
   :undoc-members:
   :member-order: bysource
