
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

Dictionaries
------------

Attribute data

.. automodule:: xradio.vis.schema
   :members: SourceInfoDict, FieldInfoDict
   :undoc-members:
   :member-order: bysource

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
