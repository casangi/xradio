Overview
========

The image schema defines how images can be represented in memory using
datasets that consist of n-dimensional arrays labeled with coordinates and
meta-information contained in attributes (see :doc:`Introduction
<../overview>`). It covers sky images (for example dirty, residual, model and
deconvolved images), point spread functions, primary beams, deconvolution
masks and the aperture plane data that images are made from (gridded
visibilities, weights and apertures). All the images of an image set that
share coordinates, for example the products of an imaging run, are held in a
single ``xarray.Dataset``, the image dataset (``img_xds``); unlike the
Measurement Set v4, images do not use an ``xarray.DataTree``. The image schema
follows the conventions of the :doc:`Measurement Set v4
<../measurement_set/overview>` schema and shares its measures (see
:ref:`measures`).

The current version of the image schema is |IMAGE_SCHEMA_VERSION|, and every
image dataset records the version it conforms to in its ``schema_version``
attribute. Datasets that XRADIO creates, and the CASA images and FITS files it
opens, carry the current version. Zarr stores written with an older version,
an empty one or none are upgraded to the current version when they are read;
a store with a newer version (written by a later XRADIO), or with any other
value that is not a semantic version, keeps it, with a warning. The schema is
under development; its versions follow the rules of the `schema versioning
<../overview.rst#Schema-Versioning>`__ section.

Reference documents consulted for the image schema design:

- casacore `Images
  <https://casacore.github.io/casacore/group__Images__module.html>`__ and
  `Coordinates
  <https://casacore.github.io/casacore/group__Coordinates__module.html>`__
  modules
- `CASA Images
  <https://casadocs.readthedocs.io/en/stable/notebooks/image_analysis.html#CASA-Images>`__
- `FITS Standard <https://fits.gsfc.nasa.gov/fits_standard.html>`__
- `FITS World Coordinate System papers I to III
  <https://fits.gsfc.nasa.gov/fits_wcs.html>`__ (world, celestial and
  spectral coordinates)
- `AIPS Memo 27: Non-linear Coordinate Systems in AIPS
  <https://library.nrao.edu/public/memos/aips/memos/AIPSM_027.pdf>`__
- The :ref:`measures <measures>` shared with the Measurement Set v4 schema

Schema Layout
-------------

Sky images have the dimensions ``(time, frequency, polarization, l, m)``,
where ``l`` and ``m`` are the projection plane coordinates towards the east
and the north, measured from the reference direction (the direction cosines
for the SIN projection). Aperture plane data have the dimensions
``(time, frequency, polarization, u, v)``, where ``u`` and ``v`` are the
Fourier conjugates of ``l`` and ``m``, in wavelengths. The polarization axis
is in Jones order (``I, Q, U, V``; ``RR, RL, LR, LL``; ``XX, XY, YX, YY``),
so that the correlations of a pair of feeds map directly onto 2x2 Jones
matrices; the readers return this order whatever the order on disk.

The data variables of an image dataset (see the :doc:`Image Schema <schema>`
for full details) are:

- ``SKY``: the image of the sky. Its ``sub_type`` attribute records the
  physical quantity held by the image, for example ``Intensity`` or
  ``SpectralIndex``.
- ``FLAG_*``: boolean images of the invalid pixels (``True`` means invalid),
  named after the image they apply to, for example ``FLAG_SKY``.
- ``MASK``: the deconvolution mask, the region(s) where the deconvolution
  algorithm is allowed to place clean components.
- ``POINT_SPREAD_FUNCTION``: the instrumental response ("dirty beam"), unity
  at the peak.
- ``PRIMARY_BEAM``: the antenna power pattern projected onto the sky.
- ``BEAM_FIT_PARAMS_*``: Gaussian fits (major axis, minor axis and position
  angle) per plane, ``BEAM_FIT_PARAMS_SKY`` of the resolution element of the
  sky image and ``BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION`` of the point spread
  function.
- ``VISIBILITY``, ``UV_SAMPLING`` and ``APERTURE``: the gridded visibilities,
  weights and weighted apertures on the aperture plane, whose Fourier
  transforms give the sky image, the point spread function and the primary
  beam (only used internally and for debugging).
- ``VISIBILITY_NORMALIZATION``, ``UV_SAMPLING_NORMALIZATION`` and
  ``APERTURE_NORMALIZATION``: the normalization of the gridded data, one value
  per ``(time, frequency, polarization)`` plane, for example the sum of
  weights (the CASA ``sumwt`` image) in ``VISIBILITY_NORMALIZATION``.

The coordinates ``time`` (typically a single value, the observation date),
``frequency`` and ``polarization`` are always present. Optional coordinates
are ``l`` and ``m`` (with the ``right_ascension`` and ``declination``, or
galactic coordinates, of every pixel when they have been computed), ``u`` and
``v``, ``beam_params_label`` (``major``, ``minor``, ``pa``) and ``velocity``,
a non-dimensional coordinate parallel to ``frequency``.

The attributes of the image dataset are:

- ``type``: always ``"image_dataset"``.
- ``schema_version``: the version of the image schema the dataset conforms
  to.
- ``data_groups``: the data groups of the dataset (see `Data Groups`_).
- ``coordinate_system_info``: the world coordinate system of sky images: the
  reference direction, the projection (for example ``SIN``) and its
  parameters, the native pole direction and the pixel coordinate
  transformation matrix (``PCi_j`` in FITS WCS).

The image variables have attributes as well, for example ``units``,
``telescope``, ``observer``, ``obsdate``, ``pointing_center``,
``object_name`` and ``user``, a dictionary of extra keywords such as leftover
FITS header cards or casacore miscinfo.
:py:func:`~xradio.image.schema.check_image` checks a dataset against the
schema, including its data groups and the data variables they reference.

Data Groups
-----------

An image dataset can contain multiple versions of a data variable. New
versions are named with the standard name followed by an underscore and a
description, for example ``SKY_DIRTY``, ``SKY_RESIDUAL`` or ``SKY_MODEL``;
flags and beam fit parameters are named after the image they apply to, for
example ``FLAG_SKY_RESIDUAL`` or ``BEAM_FIT_PARAMS_SKY_DIRTY``. To maintain
the relationship between a set of data variables, the ``data_groups``
dictionary, stored as an attribute of the image dataset, contains one or more
data group definitions. A data group maps fixed lowercase roles (``sky``,
``flag``, ``point_spread_function``, ``primary_beam``, ``mask``,
``beam_fit_params_sky``, ``beam_fit_params_point_spread_function``,
``visibility_normalization`` and the aperture plane roles) to data variable
names, and can hold a ``description`` and a creation ``date``.

Data variables can be shared between data groups or be unique to one of them.
For example, the dirty and residual images of an imaging run share their point
spread function, primary beam and mask:

.. code:: python

   img_xds.attrs["data_groups"] = {
       "dirty": {"sky": "SKY_DIRTY", "flag": "FLAG_SKY_DIRTY",
                 "point_spread_function": "POINT_SPREAD_FUNCTION",
                 "primary_beam": "PRIMARY_BEAM", "mask": "MASK"},
       "residual": {"sky": "SKY_RESIDUAL", "flag": "FLAG_SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                    "primary_beam": "PRIMARY_BEAM", "mask": "MASK"},
   }

A single CASA or FITS image opened with ``open_image`` has one data group:
``residual``, ``model`` or ``dirty`` for a residual, model or dirty image
(such as a tclean ``.residual`` or ``.model`` image, whose type is detected
from its name), and ``base`` for any other image. The ``xr_img`` accessor
selects a data group by name, returning a dataset with only the data
variables of that group:

.. code:: python

   residual_xds = img_xds.xr_img.sel(data_group_name="residual")

Storage Formats
---------------

``xradio.image.open_image`` opens CASA images, FITS files and zarr stores
lazily as an image dataset, ``load_image`` reads all or part of a CASA image
into memory (for a zarr store it returns the selected part lazily, like
``open_image``), and ``write_image`` writes an image dataset in any of the
three formats (``out_format`` ``"casa"``, the default, ``"fits"`` or
``"zarr"``) and returns the paths it wrote.

- **zarr**: a zarr store holds a whole image dataset, with all its data
  variables, data groups and attributes, and ``open_image`` reads it back as it
  was written. Image zarr stores have the extension ``.img.zarr``:
  ``write_image`` adds it to the name, replacing a bare ``.zarr``, so ``out``
  and ``out.zarr`` are both written as ``out.img.zarr`` (an existing
  ``out.zarr`` is left unchanged, with a warning).
- **CASA images** (read and written with python-casacore or casatools) and
  **FITS** files (read and written with astropy only) hold one image each.
  Several of them are opened into one dataset by passing ``open_image`` a
  dictionary that maps image types to paths, or a list of paths whose image
  types are detected from their names: of the tclean products, ``.image``,
  ``.psf``, ``.pb``, ``.mask`` and ``.sumwt`` give ``SKY``,
  ``POINT_SPREAD_FUNCTION``, ``PRIMARY_BEAM``, ``MASK`` and
  ``VISIBILITY_NORMALIZATION``, while ``.residual`` and ``.model`` give
  ``SKY_RESIDUAL`` and ``SKY_MODEL`` in their own data groups. ``write_image``
  writes every image of a data group once, to its own output: a sky image
  with its flags and beam fit parameters, and a point spread function with
  its beam fit parameters (other variables, and images the format cannot
  hold, are skipped with a warning). A single image is written to the given
  name; several images are named ``<name>.<g1>...<gn>.<role>``, where ``g1``
  to ``gn`` are, in data group order, all the data groups whose ``<role>``
  refers to the image. For
  the example above, ``write_image(img_xds, "out")`` writes ``out.dirty.sky``,
  ``out.dirty.residual.point_spread_function``,
  ``out.dirty.residual.primary_beam``, ``out.dirty.residual.mask`` and
  ``out.residual.sky``. For FITS, a ``.fits`` extension of the name stays
  last: ``out.fits`` gives ``out.dirty.sky.fits``. FITS files hold only sky
  plane images, CASA images also aperture plane images, and both a single time
  plane.
- **xarray backends**: CASA images and FITS files can also be opened with
  ``xarray.open_dataset``, through the ``xradio_casa_image`` and
  ``xradio_fits_image`` engines that XRADIO registers. Without an ``engine``
  argument, xarray picks them for CASA image tables and for ``.fits``,
  ``.fit`` or ``.fts`` files that hold an image in their primary HDU.

Translating CASA Images and FITS
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - CASA image
     - FITS file
     - Image dataset
   * - pixel values
     - primary HDU data
     - image variable (``SKY``, ``POINT_SPREAD_FUNCTION``, ...)
   * - internal mask (``True`` marks valid pixels)
     - NaN pixels
     - ``FLAG_<image>`` (``True`` marks invalid pixels)
   * - restoring beam(s)
     - ``BMAJ``, ``BMIN`` and ``BPA``, or a ``BEAMS`` table
     - ``BEAM_FIT_PARAMS_<image>``
   * - direction coordinate
     - celestial axes
     - ``l`` and ``m`` coordinates, ``coordinate_system_info``
   * - spectral coordinate
     - spectral axis
     - ``frequency`` and ``velocity`` coordinates
   * - Stokes coordinate
     - ``STOKES`` axis
     - ``polarization`` coordinate, in Jones order
   * - image type
     - ``BTYPE``
     - ``sub_type`` attribute
   * - miscinfo
     - other header cards
     - ``user`` attribute

Delving further
---------------

1. The :doc:`tutorial <tutorials/image>` demonstrates opening, checking,
   writing and creating image datasets, and the ``xr_img`` accessor.
2. The :doc:`Image Schema <schema>` page documents every data variable,
   coordinate, attribute and data group role of the schema.
3. The :doc:`API documentation <api>` describes the functions of
   ``xradio.image`` and the methods of the ``xr_img`` accessor.
