Schemas
=======

.. _correlated data datasets:

Correlated Dataset
------------------

The Measurement Set v4 schema defines a model of correlated data which can be of visibility
(:ref:`visibility-xds`) or spectrum (:ref:`spectrum-xds`) type. These are
:py:class:`xarray.Dataset` s: collections of :ref:`correlated data arrays` and
:ref:`correlated data coordinates` sharing the same dimensions, forming a comprehensive
view of correlated data. Additional metadata is defined in several :ref:`sub-datasets`
and :ref:`info dictionaries`. The visibility or spectrum datasets can have one or more
visibility, flag, weight, etc. arrays as well as :ref:`field_and_source_xds` sub-datasets,
as defined in the :ref:`data groups dictionary`.

.. _visibility-xds:

Correlated dataset: Visibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: xradio.measurement_set.schema.VisibilityXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.VisibilityXds

.. _spectrum-xds:

Correlated dataset: Spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: xradio.measurement_set.schema.SpectrumXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.SpectrumXds

.. _sub-datasets:

Sub-datasets
------------

antenna_xds
~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.AntennaXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.AntennaXds

.. _field_and_source_xds:

field_and_source_xds
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.FieldSourceXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.FieldSourceXds

.. autoclass:: xradio.measurement_set.schema.FieldSourceEphemerisXds()

   .. xradio_dataset_schema_table:: xradio.measurement_set.schema.FieldSourceEphemerisXds

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

.. _data groups dictionary:

Data Groups dictionary
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.DataGroupsDict()

   .. xradio_dict_schema_table:: xradio.measurement_set.schema.DataGroupsDict

Data Group dictionary
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.DataGroupDict()

   .. xradio_dict_schema_table:: xradio.measurement_set.schema.DataGroupDict

Creator dictionary
~~~~~~~~~~~~~~~~~~
.. autoclass:: xradio.measurement_set.schema.CreatorDict()

   .. xradio_dict_schema_table:: xradio.measurement_set.schema.CreatorDict

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

.. autoclass:: xradio.measurement_set.schema.EffectiveChannelWidthArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.EffectiveChannelWidthArray

.. autoclass:: xradio.measurement_set.schema.FrequencyCentroidArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.FrequencyCentroidArray

.. autoclass:: xradio.measurement_set.schema.PointingBeamArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.PointingBeamArray

.. autoclass:: xradio.measurement_set.schema.LocalSkyCoordArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.LocalSkyCoordArray

.. autoclass:: xradio.measurement_set.schema.PhasedArrayCoordinateAxesArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.PhasedArrayCoordinateAxesArray

.. autoclass:: xradio.measurement_set.schema.PhasedArrayElementOffsetArray()

   .. xradio_array_schema_table:: xradio.measurement_set.schema.PhasedArrayElementOffsetArray

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


.. _correlated data attributes:

Value Keys
----------

.. _scan intents:

Scan Intents
~~~~~~~~~~~~

Scan intents to be used with :py:class:`~xradio.measurement_set.schema.VisibilityXds` and
:py:class:`~xradio.measurement_set.schema.SpectrumXds`,
in the ``intents`` field of the
:py:class:`~xradio.measurement_set.schema.ObservationInfoDict`:

* ``CALIBRATE AMPLI`` : Amplitude calibration scan
* ``CALIBRATE ANTENNA PHASE`` : Requested by EVLA.
* ``CALIBRATE ANTENNA POINTING MODEL`` : Requested by EVLA.
* ``CALIBRATE ANTENNA POSITION`` : Requested by EVLA.
* ``CALIBRATE APPPHASE ACTIVE`` : Calculate and apply phasing solutions. Applicable at ALMA.
* ``CALIBRATE APPPHASE PASSIVE`` : Apply previously obtained phasing solutions. Applicable at ALMA.
* ``CALIBRATE ATMOSPHERE`` : Atmosphere calibration scan
* ``CALIBRATE BANDPASS`` : Bandpass calibration scan
* ``CALIBRATE DELAY`` : Delay calibration scan
* ``CALIBRATE DIFFGAIN`` : Enable a gain differential target type
* ``CALIBRATE FLUX`` : flux measurement scan.
* ``CALIBRATE FOCUS`` : Focus calibration scan. Z coordinate to be derived
* ``CALIBRATE FOCUS X`` : Focus calibration scan; X focus coordinate to be derived
* ``CALIBRATE FOCUS Y`` : Focus calibration scan; Y focus coordinate to be derived
* ``CALIBRATE PHASE`` : Phase calibration scan
* ``CALIBRATE POINTING`` : Pointing calibration scan
* ``CALIBRATE POL ANGLE`` :
* ``CALIBRATE POL LEAKAGE`` :
* ``CALIBRATE POLARIZATION`` : Polarization calibration scan
* ``CALIBRATE SIDEBAND RATIO`` : measure relative gains of sidebands.
* ``CALIBRATE WVR`` : Data from the water vapor radiometers (and correlation data) are used to derive their calibration parameters.
* ``DO SKYDIP`` : Skydip calibration scan
* ``MAP ANTENNA SURFACE`` : Holography calibration scan
* ``MAP PRIMARY BEAM`` : Data on a celestial calibration source are used to derive a map of the primary beam.
* ``MEASURE RFI`` : Requested by EVLA.
* ``OBSERVE CHECK SOURCE`` :
* ``OBSERVE TARGET`` : Target source scan
* ``SYSTEM CONFIGURATION`` : Requested by EVLA.
* ``TEST`` : used for development.
* ``UNSPECIFIED`` : Unspecified scan intent

Sub-scan intents to be used with :py:class:`~xradio.measurement_set.schema.VisibilityXds` and
:py:class:`~xradio.measurement_set.schema.SpectrumXds`,
in the ``intents`` field of the
:py:class:`~xradio.measurement_set.schema.ObservationInfoDict`:

* ``ON SOURCE`` : on-source measurement
* ``OFF SOURCE`` : off-source measurement
* ``MIXED`` : Pointing measurement, some antennas are on -ource, some off-source
* ``REFERENCE`` : reference measurement (used for boresight in holography).
* ``SCANNING`` : antennas are scanning.
* ``HOT`` : hot load measurement.
* ``AMBIENT`` : ambient load measurement.
* ``SIGNAL`` : Signal sideband measurement.
* ``IMAGE`` : Image sideband measurement.
* ``TEST`` : reserved for development.
* ``UNSPECIFIED`` : Unspecified

.. _spw intents:

Spectral Window Intents
~~~~~~~~~~~~~~~~~~~~~~~

Spectral window intents to be used in the attribute ``spectral_window_intent``
of the ``frequency`` coordinate of measurement sets
(:py:class:`~xradio.measurement_set.schema.VisibilityXds` and
:py:class:`~xradio.measurement_set.schema.SpectrumXds`):

* ``TEST`` : reserved for development.
* ``UNSPECIFIED`` : Unspecified SPW intent.

Note: the list is to be extended.

.. _flag bits:

Flag Bits
~~~~~~~~~

When :py:class:`~xradio.measurement_set.schema.FlagArray` is integer
data type, bits indicate flagging reason (see ``FLAG`` data variable
and ``flag_bits`` attribute in
:py:class:`~xradio.measurement_set.schema.VisibilityXds` and
:py:class:`~xradio.measurement_set.schema.SpectrumXds`). Suggested
flag bits:

* ``UNSPECIFIED_BIT`` (default bit 0): reserved for unspecified flag reason
* ``STATIC_BIT`` (default bit 1): predefined static flag list
* ``CAM_BIT`` (default bit 2): flag based on live CAM information
* ``DATA_LOST_BIT`` (default bit 3): no data was received
* ``INGEST_RFI_BIT`` (default bit 4): RFI detected in ingest
* ``PREDICTED_RFI_BIT`` (default bit 5): RFI predicted from space based pollutants
* ``CAL_RFI_BIT`` (default bit 6): RFI detected in calibration
* ``POSTPROC_BIT`` (default bit 7): some correction/postprocessing step could not be applied

These bits are derived from usage in MeerKat (see
`flags.py <https://github.com/ska-sa/katdal/blob/0840fd86ca4954168cacf4cb785eb00afef121b4/katdal/flags.py>`_).
