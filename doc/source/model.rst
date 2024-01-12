
Data Models
===========

Dimensions
----------

Fundamental dimensions of Xradio data. The shape of all arrays contained in
Xradio datasets will be defined by a mapping of dimensions to sizes -- the
"shape" of the array. For instance, a visibility data array might have an
associated channel or time step count. Use :ref:`Coordinates` to associate
dimension indicies with values such as frequencies or timestamps.

.. class:: Time
.. autodata:: xradio.vis.model.Time
.. class:: BaselineId
.. autodata:: xradio.vis.model.BaselineId
.. class:: Channel
.. autodata:: xradio.vis.model.Channel
.. class:: Polarization
.. autodata:: xradio.vis.model.Polarization
.. class:: UvwLabel
.. autodata:: xradio.vis.model.UvwLabel

Coordinates
-----------

Data model axes. Associate indices in :ref:`Dimensions` with data.

.. autoclass:: xradio.vis.model.TimeAxis

   .. xradio_array_schema_table:: xradio.vis.model.TimeAxis

.. autoclass:: xradio.vis.model.BaselineAxis

   .. xradio_array_schema_table:: xradio.vis.model.BaselineAxis

.. autoclass:: xradio.vis.model.BaselineAntennaAxis

   .. xradio_array_schema_table:: xradio.vis.model.BaselineAntennaAxis

.. autoclass:: xradio.vis.model.FrequencyAxis

   .. xradio_array_schema_table:: xradio.vis.model.FrequencyAxis

.. autoclass:: xradio.vis.model.PolarizationAxis

   .. xradio_array_schema_table:: xradio.vis.model.PolarizationAxis

.. autoclass:: xradio.vis.model.UvwLabelAxis

   .. xradio_array_schema_table:: xradio.vis.model.UvwLabelAxis


Data Arrays
-----------

Data arrays. Bulk data gathered into :ref:`Data sets`.

.. autoclass:: xradio.vis.model.VisibilityArray

   .. xradio_array_schema_table:: xradio.vis.model.VisibilityArray

.. autoclass:: xradio.vis.model.FlagArray

   .. xradio_array_schema_table:: xradio.vis.model.FlagArray

.. autoclass:: xradio.vis.model.WeightArray

   .. xradio_array_schema_table:: xradio.vis.model.WeightArray

.. autoclass:: xradio.vis.model.UvwArray

   .. xradio_array_schema_table:: xradio.vis.model.UvwArray

.. autoclass:: xradio.vis.model.TimeSamplingArray

   .. xradio_array_schema_table:: xradio.vis.model.TimeSamplingArray

.. autoclass:: xradio.vis.model.FreqSamplingArray

   .. xradio_array_schema_table:: xradio.vis.model.FreqSamplingArray


Data sets
---------

Data sets. Collects together a number of :ref:`Data arrays` and
:ref:`Coordinates` sharing the same :ref:`Dimensions` to form a comprehensive
view of the data.

.. autoclass:: xradio.vis.model.VisibilityXds

   .. xradio_dataset_schema_table:: xradio.vis.model.VisibilityXds

Information
-----------

Attribute data

.. automodule:: xradio.vis.model
   :members: SourceInfo, FieldInfo, Quantity
   :undoc-members:
   :member-order: bysource
