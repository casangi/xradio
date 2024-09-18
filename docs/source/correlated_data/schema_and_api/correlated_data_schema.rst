
Correlated data schema
======================

.. _correlated data datasets:

Main dataset
------------

Model of correlated data (visibility or spectrum) :py:class:`xarray.Dataset`: a
collection of :ref:`Correlated data arrays` and :ref:`correlated data coordinates`
sharing the same dimensions, forming a comprehensive view of correlated data. The
main dataset contains several :ref:`sub-datasets` and :ref:`info dictionaries`.
The visibility or spectrum arrays have a :ref:`field_and_source_xds` sub-dataset.

.. autoclass:: xradio.correlated_data.schema.CorrelatedDataXds()

   .. xradio_dataset_schema_table:: xradio.correlated_data.schema.CorrelatedDataXds


.. _sub-datasets:

Sub-datasets
------------

.. _field_and_source_xds:

field_and_source_xds
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.correlated_data.schema.FieldSourceXds()

   .. xradio_dataset_schema_table:: xradio.correlated_data.schema.FieldSourceXds

antenna_xds
~~~~~~~~~~~
.. autoclass:: xradio.correlated_data.schema.AntennaXds()

   .. xradio_dataset_schema_table:: xradio.correlated_data.schema.AntennaXds

pointing_xds
~~~~~~~~~~~~
.. autoclass:: xradio.correlated_data.schema.PointingXds()

   .. xradio_dataset_schema_table:: xradio.correlated_data.schema.PointingXds

weather_xds
~~~~~~~~~~~
.. autoclass:: xradio.correlated_data.schema.WeatherXds()

   .. xradio_dataset_schema_table:: xradio.correlated_data.schema.WeatherXds

system_calibration_xds
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.correlated_data.schema.SystemCalibrationXds()

   .. xradio_dataset_schema_table:: xradio.correlated_data.schema.SystemCalibrationXds

gain_curve_xds
~~~~~~~~~~~~~~
.. autoclass:: xradio.correlated_data.schema.GainCurveXds()

   .. xradio_dataset_schema_table:: xradio.correlated_data.schema.GainCurveXds

phase_calibration_xds
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.correlated_data.schema.PhaseCalibrationXds()

   .. xradio_dataset_schema_table:: xradio.correlated_data.schema.PhaseCalibrationXds

phased_array_xds
~~~~~~~~~~~~~~~~
.. autoclass:: xradio.correlated_data.schema.PhasedArrayXds()

   .. xradio_dataset_schema_table:: xradio.correlated_data.schema.PhasedArrayXds

.. _info dictionaries:

Info dictionaries
-----------------

Partition info
~~~~~~~~~~~~~~
.. autoclass:: xradio.correlated_data.schema.PartitionInfoDict()

   .. xradio_dict_schema_table:: xradio.correlated_data.schema.PartitionInfoDict

Observation info
~~~~~~~~~~~~~~~~
.. autoclass:: xradio.correlated_data.schema.ObservationInfoDict()

   .. xradio_dict_schema_table:: xradio.correlated_data.schema.ObservationInfoDict

Processor info
~~~~~~~~~~~~~~
.. autoclass:: xradio.correlated_data.schema.ProcessorInfoDict()

   .. xradio_dict_schema_table:: xradio.correlated_data.schema.ProcessorInfoDict

.. _correlated data arrays:

Data Arrays
-----------

Models of correlated data :py:class:`xarray.DataArray` s.
Bulk data gathered into :ref:`correlated data datasets`.

.. autoclass:: xradio.correlated_data.schema.VisibilityArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.VisibilityArray

.. autoclass:: xradio.correlated_data.schema.SpectrumArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.SpectrumArray

.. autoclass:: xradio.correlated_data.schema.FlagArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.FlagArray

.. autoclass:: xradio.correlated_data.schema.WeightArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.WeightArray

.. autoclass:: xradio.correlated_data.schema.UvwArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.UvwArray

.. autoclass:: xradio.correlated_data.schema.SkyCoordArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.SkyCoordArray

.. autoclass:: xradio.correlated_data.schema.TimeSamplingArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.TimeSamplingArray

.. autoclass:: xradio.correlated_data.schema.FreqSamplingArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.FreqSamplingArray

.. _correlated data coordinates:

Coordinates
-----------

Define fundamental dimensions of Xradio data, and associate them with
values. The shape of all arrays contained in Xradio datasets will be defined by
a mapping of dimensions to sizes -- the "shape" of the array. For instance, a
data array might have an associated channel or time step count. Use
:ref:`correlated data Coordinates` to associate dimension indicies with values such
as frequencies or timestamps.

.. autoclass:: xradio.correlated_data.schema.TimeArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.TimeArray

.. autoclass:: xradio.correlated_data.schema.TimeCoordArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.TimeCoordArray

.. autoclass:: xradio.correlated_data.schema.BaselineArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.BaselineArray

.. autoclass:: xradio.correlated_data.schema.BaselineAntennaNameArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.BaselineAntennaNameArray

.. autoclass:: xradio.correlated_data.schema.FrequencyArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.FrequencyArray

.. autoclass:: xradio.correlated_data.schema.PolarizationArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.PolarizationArray

.. autoclass:: xradio.correlated_data.schema.UvwLabelArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.UvwLabelArray

.. autoclass:: xradio.correlated_data.schema.QuantityArray()
                                                 
   .. xradio_array_schema_table:: xradio.correlated_data.schema.QuantityArray

.. autoclass:: xradio.correlated_data.schema.LocationArray()
                                                 
   .. xradio_array_schema_table:: xradio.correlated_data.schema.LocationArray

.. autoclass:: xradio.correlated_data.schema.TimeCalCoordArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.TimeCalCoordArray

.. autoclass:: xradio.correlated_data.schema.TimeEphemerisCoordArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.TimeEphemerisCoordArray

.. autoclass:: xradio.correlated_data.schema.TimePointingCoordArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.TimePointingCoordArray

.. autoclass:: xradio.correlated_data.schema.TimeWeatherCoordArray()

   .. xradio_array_schema_table:: xradio.correlated_data.schema.TimeWeatherCoordArray

.. _correlated data attributes:

Value Keys
----------

.. _scan intents:

Scan Intents
~~~~~~~~~~~~

Scan intents to be used with :py:class:`CorrelatedDataXds` ``.intent``:

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

Sub-scan intents to be used with :py:class:`CorrelatedDataXds` ``.sub_intent``:

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
