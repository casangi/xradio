Conventions
===========

These conventions apply to all XRADIO schemas.

Time and frequency order
------------------------

The ``time`` and ``frequency`` coordinates must always increase. Their step does not have to be constant, but a value
cannot repeat.

XRADIO handles data stored in another order as follows:

- :py:func:`~xradio.measurement_set.convert_msv2_to_processing_set` always gives increasing times, and sorts the
  frequencies of a spectral window. A repeated frequency is an error.
- The lazy readers reverse a decreasing axis on the fly: :py:func:`~xradio.measurement_set.open_msv2`, ``open_asdm``,
  :py:func:`~xradio.image.open_image` for CASA and FITS images, and :py:func:`~xradio.image.load_image` for CASA images.
- These readers raise an error for an axis that neither increases nor decreases. Convert the data, or fix the order with
  CASA: for a Measurement Set, ``split`` or ``mstransform`` with the channel ranges in frequency order. Images cannot be
  converted: fix their spectral axis, for example with CASA.
- ``frequency_system_cal`` of ``system_calibration_xds`` holds the frequencies of the spectral window, increasing, with
  the spectra reversed for a decreasing window. CASA selects, averages or reorders the channels of a spectral window
  without changing SYSCAL. For such a window, the spectra keep their order and have no ``frequency_system_cal``
  coordinate, as the Measurement Set does not give their frequencies.

Datasets created in other ways, and Processing Sets and zarr images already stored, are not checked.

Names
-----

Names are descriptive. Data variables are ``UPPER_SNAKE`` case, for example ``VISIBILITY`` and ``SKY``; coordinates
and attributes are ``lower_snake`` case, for example ``baseline_id`` and ``schema_version``. Values taken from external
standards keep their spelling: casacore frame names such as ``TOPO``, Stokes labels such as ``RR``, and astropy frame
names such as ``fk5``. The UID attributes of the observation information (``project_UID``, ...) keep the ASDM
spelling, and the image ``sub_type`` values are casacore's image types without spaces (``ColumnDensity``).

Coordinates and frames
----------------------

Coordinates follow one rule, the WCS longitude rule, in every frame. Every pair of direction coordinates is
(longitude-like, latitude-like), and each coordinate increases with its angle. A sign is never flipped inside a frame.

.. list-table::
   :header-rows: 1

   * - Frame
     - Longitude
     - Latitude
     - Pairs
   * - equatorial: sky images, Measurement Set v4
     - right ascension
     - declination
     - image (``l``, ``m``); the (``u``, ``v``) of UVW, with ``w`` towards the source
   * - alt-az: antennas, voltage patterns
     - azimuth
     - elevation
     - voltage pattern (``l``, ``m``)

- In a sky image, ``l`` increases towards the east and ``m`` towards the north, measured from the reference direction
  (see :py:class:`~xradio.image.schema.LCoordArray` and :py:class:`~xradio.image.schema.MCoordArray`); for the SIN
  projection they are the direction cosines of AIPS Memo 27. Images in the galactic frame follow the same rule with
  galactic longitude and latitude.
- In a voltage pattern in the alt-az frame, ``l`` increases towards increasing azimuth and ``m`` towards increasing
  elevation (see :py:class:`~xradio.image.schema.VoltagePatternArray`).
- Direction labels list the longitude first: ``ra, dec`` (``sky_dir_label``), ``lon, lat`` (``ellipsoid_dir_label``)
  and ``az, alt`` (``local_sky_dir_label``).

UVW
---

As in :py:class:`~xradio.measurement_set.schema.UvwArray`: ``u`` points towards increasing longitude, ``v`` lies in the
plane through the source and the pole, and ``w`` points towards the source. The baseline is antenna 1 minus antenna 2
(the "first-plus" convention of DiFX). UVW are in meters, in the frame given by their ``frame`` attribute. This
definition also determines the sign of the phase of ``VISIBILITY``.

Fourier transforms
------------------

The aperture plane coordinates ``u`` and ``v`` of images are the Fourier conjugates of ``l`` and ``m``, in wavelengths
(``units`` ``"lambda"``, see :py:class:`~xradio.image.schema.UCoordArray`). The gridded visibilities (``VISIBILITY``)
and weights (``UV_SAMPLING``) of an image dataset give the images through a Fourier transform, and the aperture
(``APERTURE``) is the Fourier transform of the primary beam (see :py:class:`~xradio.image.schema.ApertureArray`). The
schemas do not fix the sign or the normalization of these transforms.

Pixel indexing
--------------

- Image data variables have the dimensions ``(time, frequency, polarization, l, m)`` on the sky plane and
  ``(time, frequency, polarization, u, v)`` on the aperture plane, so their arrays are indexed ``[..., l, m]``. FITS
  files store the transpose, ``(m, l)`` in numpy order; the FITS reader and writer convert. Images of polynomial terms
  have ``frequency_polynomial`` or ``frequency_polynomial_moment`` in the place of ``frequency``, and
  ``time_polynomial`` in the place of ``time`` (see :py:class:`~xradio.image.schema.FrequencyPolynomialCoordArray`).
- Pixel indices keep their FITS meaning: the readers keep the sign of ``CDELT``, so in the usual orientation ``l``
  decreases with the pixel index, as right ascension does, and the ``make_empty_*`` factories make sky images that
  way. Going between ``(l, m)`` and ``(m, l)`` arrays is a transpose, never a flip.
- Coordinate values belong to the pixel centres: along a linear axis, the value at index ``i`` is
  ``crval + (i - crpix) * cdelt``, where ``crpix`` is the reference pixel (``l`` and ``m`` are 0 there).
- Reference pixels are counted from 0. FITS ``CRPIX`` counts from 1, so the FITS reader subtracts 1 and the writer adds
  1.
- The reference pixel is where ``l = m = 0`` (``u = v = 0``). It may lie outside the image and need not be an integer.
  The ``make_empty_*`` factories put it at index ``n // 2`` of an axis with ``n`` pixels.

Sky image pixels and FITS WCS
-----------------------------

:py:meth:`~xradio.image.image_xds.ImageXdsAccessor.get_fits_header` gives the FITS header that
:py:func:`~xradio.image.write_image` writes for a sky image, and
:py:meth:`~xradio.image.image_xds.ImageXdsAccessor.get_wcs` its astropy WCS. Both describe the pixels as the FITS file
stores them, the transpose of the dataset's (see `Pixel indexing`_):

- The celestial WCS has pixel axis 0 along ``l`` and pixel axis 1 along ``m``: ``pixel_to_world(i_l, i_m)`` takes the
  dataset's ``l`` and ``m`` indices and gives positions in the image's frame (``RA``/``DEC`` axes for ``fk4``,
  ``fk5`` and ``icrs``, ``GLON``/``GLAT`` for ``galactic``). Its ``array_shape`` is ``(m, l)``, the order of FITS
  data in numpy, so transpose a sky plane to use it with the WCS.
- The WCS of all the axes of the file (``get_wcs(celestial=False)``) has the axes ``l``, ``m``, ``STOKES`` and
  ``FREQ``, and no time axis: a FITS file holds one time plane, whose time is in ``DATE-OBS`` and ``MJD-OBS``.
- A FITS ``STOKES`` axis must be linear in the FITS polarization codes, so FITS files store four correlations in
  canonical order (see `Polarization`_), RR, RL, LR, LL (XX, XY, YX, YY), as RR, LL, RL, LR (XX, YY, XY, YX). The
  readers restore the canonical order; the header and the full WCS follow the order of the file.

Units and time
--------------

- Quantities and measures give their units in a ``units`` attribute. The image ``l`` and ``m`` coordinates have no
  ``units``: they are in radians (dimensionless direction cosines for the SIN projection). The unit values the schemas
  fix are ``s``, ``Hz``, ``m``, ``rad``, ``m/s`` and, for Doppler shifts, ``ratio``, plus ``K``, ``K/Jy``, ``Pa``,
  ``/m^2`` and ``dimensionless`` in the Measurement Set. Image ``u`` and ``v`` are in wavelengths (``lambda``) and
  image times in days (``d``) or seconds. Data values carry their own units, for example ``Jy`` for ``VISIBILITY``.
- Times are floating point numbers since an epoch, interpreted as :py:class:`astropy.time.Time` with the ``units``,
  ``format`` and ``scale`` attributes: first as a quantity in ``units``, then in ``format``
  (see :py:class:`~xradio.schema.measures.TimeArray`). Measurement Set v4 times are in seconds, by default in format
  ``unix`` and scale ``utc``; image times are by default in days, format ``mjd`` and scale ``utc``.
- Time scales are astropy's: ``tai``, ``tcb``, ``tcg``, ``tdb``, ``tt``, ``ut1`` and ``utc``. Formats are ``unix``,
  ``mjd``, ``cxcsec`` and ``gps``. The image readers translate casacore epoch references (and FITS ``TIMESYS``) with
  :py:data:`~xradio.schema.measures.CASACORE_EPOCH_REF_TO_SCALE`; converting or opening a Measurement Set v2
  translates only ``UTC`` and ``TAI`` and keeps other references as they are.

Measures and frames
-------------------

Measures follow astropy's conventions (see :ref:`measures`). Frame values are astropy's names, except where xradio
does not translate a casacore frame: those keep their casacore names (``TOPO``, ``BARY``, ...). The ``observer`` of a
spectral coordinate stores only the name of the frame the observer is stationary relative to, where astropy's
``observer`` is a full coordinate. The table lists the values and where each comes from; the frame of an image's
spectral axis and of its telescope position use casacore names.

.. list-table::
   :header-rows: 1

   * - Measure
     - Values
     - From
   * - sky coordinates (``frame``)
     - ``icrs``, ``fk5``, ``fk4``, ``galactic``, ``altaz``, ... (astropy frame names, except ``fk4noterms``, which
       astropy spells ``fk4noeterms``)
     - astropy
   * - UVW (``frame``)
     - ``fk5``, ``icrs``; ``APP`` (apparent geocentric, WSRT)
     - astropy; casacore
   * - spectral coordinates (``observer``)
     - ``gcrs``, ``icrs``, ``hcrs``, ``lsrk``, ``lsrd``, ``lsr``; ``REST``, ``BARY``, ``TOPO``, ``GALACTO``, ``LGROUP``,
       ``CMB``
     - astropy; casacore
   * - image spectral axis (``frame``)
     - ``REST``, ``LSRK``, ``LSRD``, ``BARY``, ``GEO``, ``TOPO``, ``GALACTO``, ``LGROUP``, ``CMB``; the
       ``make_empty_*`` factories can also give ``icrs``, ``hcrs`` or ``lsr``
     - casacore; astropy
   * - Earth locations (``frame``)
     - ``ITRS`` (translated from casacore's ``ITRF``)
     - astropy
   * - image telescope position (``frame``)
     - ``ITRF``
     - casacore
   * - native pole of an image projection (``frame``)
     - ``NATIVE_PROJECTION``
     - xradio
   * - Doppler (``doppler_type``)
     - ``radio``, ``optical``, ``z``, ``ratio``, ``true``, ``relativistic``, ``beta``, ``gamma``
     - casacore (astropy's ``SpectralCoord`` has ``radio``, ``optical`` and ``relativistic``)

The Measurement Set converters translate casacore direction references with
:py:data:`~xradio.schema.measures.CASACORE_DIRECTION_TO_FRAME` (``AZELGEO`` to ``altaz``, ``J2000`` to ``fk5``,
``ICRS`` to ``icrs``); the CASA image reader translates ``J2000`` to ``fk5`` (equinox ``j2000.0``), ``B1950`` to
``fk4`` (``b1950.0``), and ``GALACTIC`` and ``ICRS`` to ``galactic`` and ``icrs``. Spectral frames translate with
:py:data:`~xradio.schema.measures.CASACORE_TO_OBSERVER`: ``LSRK`` to ``lsrk``, ``LSRD`` to ``lsrd`` and ``GEO`` to
``gcrs``; the other frames keep their casacore names (see
:py:data:`~xradio.schema.measures.AllowedSpectralCoordFrames`).

Polarization
------------

Polarization labels are the casacore ``Stokes`` names (:py:data:`~xradio.schema.polarization.CASACORE_STOKES_TYPES`).
The canonical order is I, Q, U, V; RR, RL, LR, LL; XX, XY, YX, YY
(:py:data:`~xradio.schema.polarization.CANONICAL_POLARIZATION_ORDER`), so that the correlations of a pair of feeds
are in 2x2 Jones matrix order. Images keep this order: the readers return it, the ``make_empty_*`` factories require
it and :py:func:`~xradio.image.schema.check_image` reports any other (with
:py:func:`~xradio.schema.polarization.check_polarization_order`). The MSv2 converter keeps the correlation order of
the MSv2 (its ``CORR_TYPE``), and nothing checks the order of Measurement Set v4 correlations.

Flags
-----

A flag that is ``True`` means that the data are invalid. Image flags
(:py:class:`~xradio.image.schema.FlagArray`) are boolean. Measurement Set v4 flags
(:py:class:`~xradio.measurement_set.schema.FlagArray`) are boolean or unsigned integers: nonzero means invalid, and the
bits follow the ``flag_bits`` attribute (see :ref:`flag bits`). The deconvolution mask of an image
(:py:class:`~xradio.image.schema.MaskArray`) is the opposite: nonzero or ``True`` means selected. casacore image masks
are ``True`` for valid pixels, so the CASA image reader and writer invert them; the FITS reader flags ``NaN`` pixels
(unless it is called with ``compute_mask=False``).

Beam parameters
---------------

Gaussian beam fit parameters (:py:class:`~xradio.image.schema.BeamFitParamsArray`) are labelled ``major``, ``minor``
and ``pa``, in this order, and are in radians. As in FITS and casacore, ``major`` and ``minor`` are the full widths at
half maximum (FWHM) of the major and minor axes, and ``pa`` is the position angle of the major axis, measured from
north through east. casacore's ``positionangle`` is ``pa``; the FITS header cards ``BMAJ``, ``BMIN`` and ``BPA`` are
in degrees.

Stores
------

Zarr stores of processing sets end in ``.ps.zarr`` and those of images in ``.img.zarr``.
