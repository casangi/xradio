Image Schema
============

.. _image dataset:

Image Dataset
-------------

The image schema defines a model of images as an :py:class:`xarray.Dataset`:
a collection of :ref:`image data arrays` and :ref:`image coordinates` that
share dimensions. A dataset can hold multiple versions of an image type
(for example ``SKY``, ``SKY_DECONVOLVED``, ``SKY_MODEL``), distinguished by an
underscore separated suffix of the canonical variable name. The
:ref:`image data groups dictionary` groups the variables that belong together
and maps logical roles to concrete variable names.

Sky plane images are defined on the ``(l, m)`` direction cosine dimensions,
aperture plane data (gridded visibilities, gridded weights and apertures) on
the conjugate ``(u, v)`` dimensions.

In addition to :py:func:`xradio.schema.check.check_dataset`, image datasets
can be checked with :py:func:`xradio.image.schema.check_image`, which also
validates the data groups and the data variables they reference.

.. autofunction:: xradio.image.schema.check_image

.. autoclass:: xradio.image.schema.ImageXds()

   .. xradio_dataset_schema_table:: xradio.image.schema.ImageXds

.. _image data groups dictionary:

Data groups
-----------

.. autoclass:: xradio.image.schema.DataGroupsDict()

   .. xradio_dict_schema_table:: xradio.image.schema.DataGroupsDict

.. autoclass:: xradio.image.schema.DataGroupDict()

   .. xradio_dict_schema_table:: xradio.image.schema.DataGroupDict

.. _image data arrays:

Data Arrays
-----------

.. autoclass:: xradio.image.schema.SkyArray()

   .. xradio_array_schema_table:: xradio.image.schema.SkyArray

.. autoclass:: xradio.image.schema.FlagArray()

   .. xradio_array_schema_table:: xradio.image.schema.FlagArray

.. autoclass:: xradio.image.schema.BeamFitParamsArray()

   .. xradio_array_schema_table:: xradio.image.schema.BeamFitParamsArray

.. autoclass:: xradio.image.schema.MaskArray()

   .. xradio_array_schema_table:: xradio.image.schema.MaskArray

.. autoclass:: xradio.image.schema.PrimaryBeamArray()

   .. xradio_array_schema_table:: xradio.image.schema.PrimaryBeamArray

.. autoclass:: xradio.image.schema.PointSpreadFunctionArray()

   .. xradio_array_schema_table:: xradio.image.schema.PointSpreadFunctionArray

.. autoclass:: xradio.image.schema.VisibilityNormalizationArray()

   .. xradio_array_schema_table:: xradio.image.schema.VisibilityNormalizationArray

.. autoclass:: xradio.image.schema.VisibilityArray()

   .. xradio_array_schema_table:: xradio.image.schema.VisibilityArray

.. autoclass:: xradio.image.schema.UvSamplingArray()

   .. xradio_array_schema_table:: xradio.image.schema.UvSamplingArray

.. autoclass:: xradio.image.schema.UvSamplingNormalizationArray()

   .. xradio_array_schema_table:: xradio.image.schema.UvSamplingNormalizationArray

.. autoclass:: xradio.image.schema.ApertureArray()

   .. xradio_array_schema_table:: xradio.image.schema.ApertureArray

.. autoclass:: xradio.image.schema.ApertureNormalizationArray()

   .. xradio_array_schema_table:: xradio.image.schema.ApertureNormalizationArray

.. _image coordinates:

Coordinates
-----------

.. autoclass:: xradio.image.schema.TimeCoordArray()

   .. xradio_array_schema_table:: xradio.image.schema.TimeCoordArray

.. autoclass:: xradio.image.schema.FrequencyCoordArray()

   .. xradio_array_schema_table:: xradio.image.schema.FrequencyCoordArray

.. autoclass:: xradio.image.schema.VelocityCoordArray()

   .. xradio_array_schema_table:: xradio.image.schema.VelocityCoordArray

.. autoclass:: xradio.image.schema.LCoordArray()

   .. xradio_array_schema_table:: xradio.image.schema.LCoordArray

.. autoclass:: xradio.image.schema.MCoordArray()

   .. xradio_array_schema_table:: xradio.image.schema.MCoordArray

.. autoclass:: xradio.image.schema.UCoordArray()

   .. xradio_array_schema_table:: xradio.image.schema.UCoordArray

.. autoclass:: xradio.image.schema.VCoordArray()

   .. xradio_array_schema_table:: xradio.image.schema.VCoordArray

.. autoclass:: xradio.image.schema.BeamParamsLabelCoordArray()

   .. xradio_array_schema_table:: xradio.image.schema.BeamParamsLabelCoordArray

.. _image measures and info dictionaries:

Measures and info dictionaries
------------------------------

The measures used by the image schema (sky coordinates, spectral coordinates,
quantities) are shared with the measurement set schema and are defined in
:py:mod:`xradio.schema.measures`. The following are specific to images:

.. autoclass:: xradio.image.schema.TimeMeasureArray()

   .. xradio_array_schema_table:: xradio.image.schema.TimeMeasureArray

.. autoclass:: xradio.image.schema.SkyDirectionArray()

   .. xradio_array_schema_table:: xradio.image.schema.SkyDirectionArray

.. autoclass:: xradio.image.schema.TelescopeLocationArray()

   .. xradio_array_schema_table:: xradio.image.schema.TelescopeLocationArray

.. autoclass:: xradio.image.schema.NativePoleDirectionArray()

   .. xradio_array_schema_table:: xradio.image.schema.NativePoleDirectionArray

.. autoclass:: xradio.image.schema.TelescopeDict()

   .. xradio_dict_schema_table:: xradio.image.schema.TelescopeDict

.. autoclass:: xradio.image.schema.CoordinateSystemInfoDict()

   .. xradio_dict_schema_table:: xradio.image.schema.CoordinateSystemInfoDict

.. autoclass:: xradio.image.schema.UserDict()

   .. xradio_dict_schema_table:: xradio.image.schema.UserDict
