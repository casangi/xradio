
:tocdepth: 3
          
Data Models
===========

Dimensions
----------

Fundamental dimensions of Xradio data. The shape of all arrays contained in
Xradio datasets will be defined by a mapping of dimensions to sizes -- the
"shape" of the array. For instance, a visibility data array might have an
associated channel or time step count. Use :ref:`Axes` to associate dimension
indicies with values such as frequencies or timestamps.

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

Axes
----

Data model axes. Associate indices in :ref:`Dimensions` with data.

.. automodule:: xradio.vis.model
   :members: TimeAxis, BaselineAxis, BaselineAntennaAxis, FrequencyAxis, PolarizationAxis, UvwLabelAxis
   :undoc-members:
   :member-order: bysource

Data Arrays
-----------

Data arrays. Bulk data gathered into :ref:`Data sets`.

.. automodule:: xradio.vis.model
   :members: VisibilityArray, FlagArray, WeightArray, UvwArray, TimeSamplingArray, FreqSamplingArray
   :undoc-members:
   :member-order: bysource


Data sets
---------

Data sets. Collects together a number of :ref:`Data arrays` and :ref:`Axes`
sharing the same :ref:`Dimensions` to form a comprehensive view of the data.

.. automodule:: xradio.vis.model
   :members: VisibilityXds, SpectralCoordXds, AntennaXds, PointingXds, SourceXds, PhasedArrayXds
   :undoc-members:
   :member-order: bysource

Information
-----------

Attribute data

.. automodule:: xradio.vis.model
   :members: FieldInfo
   :undoc-members:
   :member-order: bysource
