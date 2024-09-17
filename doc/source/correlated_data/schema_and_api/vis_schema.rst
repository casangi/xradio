
Visibility schema
=================

.. _visibility datasets:

Main dataset
------------

Model of visibility (or spectrum) :py:class:`xarray.Dataset`: A collection of
:ref:`Visibility arrays` and :ref:`visibility Coordinates` sharing the same
dimensions, forming a comprehensive view of visibility data. The main dataset
contains several :ref:`sub-datasets` and :ref:`info dictionaries`. The
visibility or spectrum arrays have a :ref:`field_and_source_xds` sub-dataset.

.. autoclass:: xradio.vis.schema.VisibilityXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.VisibilityXds


.. _sub-datasets:

Sub-datasets
------------

.. _field_and_source_xds:

field_and_source_xds
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.vis.schema.FieldSourceXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.FieldSourceXds

antenna_xds
~~~~~~~~~~~
.. autoclass:: xradio.vis.schema.AntennaXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.AntennaXds

pointing_xds
~~~~~~~~~~~~
.. autoclass:: xradio.vis.schema.PointingXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.PointingXds

weather_xds
~~~~~~~~~~~
.. autoclass:: xradio.vis.schema.WeatherXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.WeatherXds

system_calibration_xds
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.vis.schema.SystemCalibrationXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.SystemCalibrationXds

gain_curve_xds
~~~~~~~~~~~~~~
.. autoclass:: xradio.vis.schema.GainCurveXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.GainCurveXds

phase_calibration_xds
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.vis.schema.PhaseCalibrationXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.PhaseCalibrationXds

phased_array_xds
~~~~~~~~~~~~~~~~
.. autoclass:: xradio.vis.schema.PhasedArrayXds()

   .. xradio_dataset_schema_table:: xradio.vis.schema.PhasedArrayXds

.. _info dictionaries:

Info dictionaries
-----------------

Partition info
~~~~~~~~~~~~~~
.. autoclass:: xradio.vis.schema.PartitionInfoDict()

   .. xradio_dict_schema_table:: xradio.vis.schema.PartitionInfoDict

Observation info
~~~~~~~~~~~~~~~~
.. autoclass:: xradio.vis.schema.ObservationInfoDict()

   .. xradio_dict_schema_table:: xradio.vis.schema.ObservationInfoDict

Processor info
~~~~~~~~~~~~~~
.. autoclass:: xradio.vis.schema.ProcessorInfoDict()

   .. xradio_dict_schema_table:: xradio.vis.schema.ProcessorInfoDict

.. _visibility arrays:

Data Arrays
-----------

Models of visibility :py:class:`xarray.DataArray` s.
Bulk data gathered into :ref:`Visibility Datasets`.

.. autoclass:: xradio.vis.schema.VisibilityArray()

   .. xradio_array_schema_table:: xradio.vis.schema.VisibilityArray

.. autoclass:: xradio.vis.schema.SpectrumArray()

   .. xradio_array_schema_table:: xradio.vis.schema.SpectrumArray

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

.. autoclass:: xradio.vis.schema.TimeCoordArray()

   .. xradio_array_schema_table:: xradio.vis.schema.TimeCoordArray

.. autoclass:: xradio.vis.schema.BaselineArray()

   .. xradio_array_schema_table:: xradio.vis.schema.BaselineArray

.. autoclass:: xradio.vis.schema.BaselineAntennaNameArray()

   .. xradio_array_schema_table:: xradio.vis.schema.BaselineAntennaNameArray

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

Value Keys
----------

.. _scan intents:

Scan Intents
~~~~~~~~~~~~

Scan intents to be used with :py:class:`VisibilityXds` ``.intent``:

* CALIBRATE AMPLI : Amplitude calibration scan
* CALIBRATE ANTENNA PHASE : Requested by EVLA.
* CALIBRATE ANTENNA POINTING MODEL : Requested by EVLA.
* CALIBRATE ANTENNA POSITION : Requested by EVLA.
* CALIBRATE APPPHASE ACTIVE : Calculate and apply phasing solutions. Applicable at ALMA.
* CALIBRATE APPPHASE PASSIVE : Apply previously obtained phasing solutions. Applicable at ALMA.
* CALIBRATE ATMOSPHERE : Atmosphere calibration scan
* CALIBRATE BANDPASS : Bandpass calibration scan
* CALIBRATE DELAY : Delay calibration scan
* CALIBRATE DIFFGAIN : Enable a gain differential target type
* CALIBRATE FLUX : flux measurement scan.
* CALIBRATE FOCUS : Focus calibration scan. Z coordinate to be derived
* CALIBRATE FOCUS X : Focus calibration scan; X focus coordinate to be derived
* CALIBRATE FOCUS Y : Focus calibration scan; Y focus coordinate to be derived
* CALIBRATE PHASE : Phase calibration scan
* CALIBRATE POINTING : Pointing calibration scan
* CALIBRATE POL ANGLE :
* CALIBRATE POL LEAKAGE :
* CALIBRATE POLARIZATION : Polarization calibration scan
* CALIBRATE SIDEBAND RATIO : measure relative gains of sidebands.
* CALIBRATE WVR : Data from the water vapor radiometers (and correlation data) are used to derive their calibration parameters.
* DO SKYDIP : Skydip calibration scan
* MAP ANTENNA SURFACE : Holography calibration scan
* MAP PRIMARY BEAM : Data on a celestial calibration source are used to derive a map of the primary beam.
* MEASURE RFI : Requested by EVLA.
* OBSERVE CHECK SOURCE :
* OBSERVE TARGET : Target source scan
* SYSTEM CONFIGURATION : Requested by EVLA.
* TEST : used for development.
* UNSPECIFIED : Unspecified scan intent

Sub-scan intents to be used with :py:class:`VisibilityXds` ``.sub_intent``:

* ON SOURCE : on-source measurement
* OFF SOURCE : off-source measurement
* MIXED : Pointing measurement, some antennas are on -ource, some off-source
* REFERENCE : reference measurement (used for boresight in holography).
* SCANNING : antennas are scanning.
* HOT : hot load measurement.
* AMBIENT : ambient load measurement.
* SIGNAL : Signal sideband measurement.
* IMAGE : Image sideband measurement.
* TEST : reserved for development.
* UNSPECIFIED : Unspecified
