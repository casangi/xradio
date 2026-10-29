Primary Beam Schema
===================

.. _zpc dataset:

Zernike Polynomial Coefficient Dataset
--------------------------------------

The Zernike polynomial coefficient dataset (``zpc_xds``) models the complex
aperture illumination of an antenna, or of a group of identical antennas, for
each correlation of a feed pair, as the coefficients of a Zernike polynomial
expansion over the aperture, one set per frequency. Its ``frequency`` and
``polarization`` coordinates follow the measurement set v4 schema.

In addition to :py:func:`xradio.schema.check.check_dataset`, datasets can be
checked with :py:func:`xradio.primary_beam.schema.check_zpc`.

.. autofunction:: xradio.primary_beam.schema.check_zpc

.. autoclass:: xradio.primary_beam.schema.ZernikePolynomialCoefficientsXds()

   .. xradio_dataset_schema_table:: xradio.primary_beam.schema.ZernikePolynomialCoefficientsXds

.. autodata:: xradio.primary_beam.schema.AllowedZernikeBases

.. autodata:: xradio.primary_beam.schema.AllowedApertureFrames

.. autoclass:: xradio.primary_beam.schema.OriginDict()

   .. xradio_dict_schema_table:: xradio.primary_beam.schema.OriginDict

Coordinates and data variables
------------------------------

.. autoclass:: xradio.primary_beam.schema.FrequencyCoordArray()

   .. xradio_array_schema_table:: xradio.primary_beam.schema.FrequencyCoordArray

.. autoclass:: xradio.primary_beam.schema.CoefficientCoordArray()

   .. xradio_array_schema_table:: xradio.primary_beam.schema.CoefficientCoordArray

.. autoclass:: xradio.primary_beam.schema.ZpcArray()

   .. xradio_array_schema_table:: xradio.primary_beam.schema.ZpcArray

.. autoclass:: xradio.primary_beam.schema.EtaArray()

   .. xradio_array_schema_table:: xradio.primary_beam.schema.EtaArray
