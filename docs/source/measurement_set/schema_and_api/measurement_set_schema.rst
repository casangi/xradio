
Measurement Set Schema v4.0.0
=============================

.. _correlated data datasets:

Correlated Dataset
------------------

Model of correlated data (visibility or spectrum) :py:class:`xarray.Dataset`: a
collection of :ref:`Correlated data arrays` and :ref:`correlated data coordinates`
sharing the same dimensions, forming a comprehensive view of correlated data. The
main dataset contains several :ref:`sub-datasets` and :ref:`info dictionaries`.
The visibility or spectrum arrays have a :ref:`field_and_source_xds` sub-dataset.

.. autoclass:: xradio.measurement_set.schema.VisibilityXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.VisibilityXds

.. autoclass:: xradio.measurement_set.schema.SpectrumXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.SpectrumXds

.. _sub-datasets:

Sub-datasets
------------

.. _field_and_source_xds:

field_and_source_xds
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.FieldSourceXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.FieldSourceXds

.. autoclass:: xradio.measurement_set.schema.FieldSourceEphemerisXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.FieldSourceEphemerisXds

antenna_xds
~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.AntennaXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.AntennaXds

pointing_xds
~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.PointingXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.PointingXds

weather_xds
~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.WeatherXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.WeatherXds

system_calibration_xds
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.SystemCalibrationXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.SystemCalibrationXds

gain_curve_xds
~~~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.GainCurveXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.GainCurveXds

phase_calibration_xds
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.PhaseCalibrationXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.PhaseCalibrationXds

phased_array_xds
~~~~~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.PhasedArrayXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.PhasedArrayXds

.. _info dictionaries:

Info dictionaries
-----------------

Observation info
~~~~~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.ObservationInfoDict()

   .. xradio_dict_schema_table:: xradio.measurement_set.schema.ObservationInfoDict

Processor info
~~~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.ProcessorInfoDict()

   .. xradio_dict_schema_table:: xradio.measurement_set.schema.ProcessorInfoDict

.. _correlated data arrays:

Data Arrays
-----------

Models of correlated data :py:class:`xarray.DataArray` s.
Bulk data gathered into :ref:`correlated data datasets`.

.. autoclass:: xradio.measurement_set.schema.VisibilityArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.VisibilityArray

.. autoclass:: xradio.measurement_set.schema.SpectrumArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.SpectrumArray

.. autoclass:: xradio.measurement_set.schema.FlagArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.FlagArray

.. autoclass:: xradio.measurement_set.schema.WeightArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.WeightArray

.. autoclass:: xradio.measurement_set.schema.UvwArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.UvwArray

.. autoclass:: xradio.measurement_set.schema.TimeSamplingArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.TimeSamplingArray

.. _correlated data coordinates:

Coordinates
-----------

Define fundamental dimensions of Xradio data, and associate them with
values. The shape of all arrays contained in Xradio datasets will be defined by
a mapping of dimensions to sizes -- the "shape" of the array. For instance, a
data array might have an associated channel or time step count. Use
:ref:`correlated data Coordinates` to associate dimension indicies with values such
as frequencies or timestamps.

.. autoclass:: xradio.measurement_set.schema.TimeCoordArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.TimeCoordArray

.. autoclass:: xradio.measurement_set.schema.BaselineArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.BaselineArray

.. autoclass:: xradio.measurement_set.schema.AntennaNameArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.AntennaNameArray

.. autoclass:: xradio.measurement_set.schema.BaselineAntennaNameArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.BaselineAntennaNameArray

.. autoclass:: xradio.measurement_set.schema.FrequencyArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.FrequencyArray

.. autoclass:: xradio.measurement_set.schema.PolarizationArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.PolarizationArray

.. autoclass:: xradio.measurement_set.schema.UvwLabelArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.UvwLabelArray

.. autoclass:: xradio.measurement_set.schema.TimeInterpolatedCoordArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.TimeInterpolatedCoordArray

.. autoclass:: xradio.measurement_set.schema.TimeSystemCalCoordArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.TimeSystemCalCoordArray

.. autoclass:: xradio.measurement_set.schema.FrequencySystemCalArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.FrequencySystemCalArray

.. autoclass:: xradio.measurement_set.schema.TimeEphemerisCoordArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.TimeEphemerisCoordArray

.. autoclass:: xradio.measurement_set.schema.TimePointingCoordArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.TimePointingCoordArray

.. autoclass:: xradio.measurement_set.schema.TimeWeatherCoordArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.TimeWeatherCoordArray


.. _correlated data measures:

Measure arrays
--------------

.. autoclass:: xradio.measurement_set.schema.TimeArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.TimeArray

.. autoclass:: xradio.measurement_set.schema.SpectralCoordArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.SpectralCoordArray

.. autoclass:: xradio.measurement_set.schema.SkyCoordArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.SkyCoordArray

.. autoclass:: xradio.measurement_set.schema.PointingBeamArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.PointingBeamArray

.. autoclass:: xradio.measurement_set.schema.LocalSkyCoordArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.LocalSkyCoordArray

.. autoclass:: xradio.measurement_set.schema.LocationArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.LocationArray

.. autoclass:: xradio.measurement_set.schema.EllipsoidPosLocationArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.EllipsoidPosLocationArray

.. autoclass:: xradio.measurement_set.schema.DopplerArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.DopplerArray


.. _correlated data quantities:

Quantity arrays
---------------

.. autoclass:: xradio.measurement_set.schema.QuantityInSecondsArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInSecondsArray

.. autoclass:: xradio.measurement_set.schema.QuantityInHertzArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInHertzArray

.. autoclass:: xradio.measurement_set.schema.QuantityInMetersArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInMetersArray

.. autoclass:: xradio.measurement_set.schema.QuantityInMetersPerSecondArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInMetersPerSecondArray

.. autoclass:: xradio.measurement_set.schema.QuantityInRadiansArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInRadiansArray

.. autoclass:: xradio.measurement_set.schema.QuantityInKelvinArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInKelvinArray

.. autoclass:: xradio.measurement_set.schema.QuantityInKelvinPerJanskyArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInKelvinPerJanskyArray

.. autoclass:: xradio.measurement_set.schema.QuantityInPascalArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInPascalArray

.. autoclass:: xradio.measurement_set.schema.QuantityInPerSquareMetersArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInPerSquareMetersArray

.. _correlated data attributes:

Value Keys
----------

.. _scan intents:

Scan Intents
~~~~~~~~~~~~

Scan intents to be used with :py:class:`VisibilityXds` and :py:class:`SpectrumXds` ``.intent``:

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

Sub-scan intents to be used with :py:class:`VisibilityXds` and :py:class:`SpectrumXds` ``.sub_intent``:

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
