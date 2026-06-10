"""Build a default CASA-image coordinate-system record.

When a CASA image is created from a bare shape (``image(name, shape=...)``)
without an explicit coordinate system, casacore fills in a default coordinate
system (``CoordinateUtil::defaultCoords``). We reproduce that record here, in
pure Python, so that an image created on top of ``casacoretables`` carries a
``coords`` keyword that the xradio CASA-image reader can parse.

The record produced below mirrors, field for field, what
``casacore.images.image(name, shape=...).info()["coordinates"]`` returns for a
4-D float image, with the axis-dependent fields (sizes, stokes labels, spectral
WCS) parameterized by the requested shape.
"""

import numpy as np

# Default Stokes labels, in the canonical casacore order.
_DEFAULT_STOKES = ["I", "Q", "U", "V"]


def build_default_coords(np_shape) -> dict:
    """Return a default CASA coordinate-system record for a 4-D image.

    Parameters
    ----------
    np_shape : sequence of int
        Image shape in numpy (C) order. A 4-D shape ``(nfreq, npol, nm, nl)``
        is expected (this matches how casacore lays out a default
        direction/stokes/spectral image: the numpy-fastest axes are the
        direction axes). Shapes with fewer than 4 dimensions are padded with
        leading length-1 axes.

    Returns
    -------
    dict
        A record equivalent to the ``coords`` table keyword of a default CASA
        image, suitable for ``table.putkeyword("coords", ...)``.
    """
    shape = [int(s) for s in np_shape]
    if len(shape) < 4:
        shape = [1] * (4 - len(shape)) + shape
    nfreq, npol, nm, nl = shape[0], shape[1], shape[2], shape[3]

    npol = max(int(npol), 1)
    stokes = (_DEFAULT_STOKES * ((npol // len(_DEFAULT_STOKES)) + 1))[:npol]

    # Spectral defaults matching casacore's default spectral coordinate.
    crpix = float(nfreq) / 2.0
    cdelt = 1000.0
    crval = 1415000000.0
    restfreq = 1420405751.7860003

    coords = {
        "telescope": "ALMA",
        "telescopeposition": {
            "m0": {"unit": "rad", "value": -1.1825465955049892},
            "m1": {"unit": "rad", "value": -0.39941498692627375},
            "m2": {"unit": "m", "value": 6379946.013264429},
            "refer": "ITRF",
            "type": "position",
        },
        "observer": "Karl Jansky",
        "obsdate": {
            "m0": {"unit": "d", "value": 51544.00000000116},
            "refer": "UTC",
            "type": "epoch",
        },
        "pointingcenter": {"initial": True, "value": np.array([0.0, 0.0])},
        "direction0": {
            "_axes_sizes": np.array([nl, nm], dtype=np.int32),
            "_image_axes": np.array([2, 3], dtype=np.int32),
            "axes": ["Right Ascension", "Declination"],
            "cdelt": np.array([-1.0, 1.0]),
            "conversionSystem": "J2000",
            "crpix": np.array([0.0, 0.0]),
            "crval": np.array([0.0, 0.0]),
            "latpole": 0.0,
            "longpole": 180.0,
            "pc": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "projection": "SIN",
            "projection_parameters": np.array([0.0, 0.0]),
            "system": "J2000",
            "units": ["'", "'"],
        },
        "stokes1": {
            "_axes_sizes": np.array([npol], dtype=np.int32),
            "_image_axes": np.array([1], dtype=np.int32),
            "axes": ["Stokes"],
            "cdelt": np.array([1.0]),
            "crpix": np.array([0.0]),
            "crval": np.array([1.0]),
            "pc": np.array([[1.0]]),
            "stokes": stokes,
        },
        "spectral2": {
            "_axes_sizes": np.array([nfreq], dtype=np.int32),
            "_image_axes": np.array([0], dtype=np.int32),
            "formatUnit": "",
            "name": "Frequency",
            "nativeType": 0,
            "restfreq": restfreq,
            "restfreqs": np.array([restfreq]),
            "system": "LSRK",
            "unit": "Hz",
            "velType": 0,
            "velUnit": "km/s",
            "version": 2,
            "waveUnit": "mm",
            "wcs": {
                "cdelt": cdelt,
                "crpix": crpix,
                "crval": crval,
                "ctype": "FREQ",
                "pc": 1.0,
            },
        },
        "pixelmap0": np.array([0, 1], dtype=np.int32),
        "pixelmap1": np.array([2], dtype=np.int32),
        "pixelmap2": np.array([3], dtype=np.int32),
        "pixelreplace0": np.array([0.0, 0.0]),
        "pixelreplace1": np.array([0.0]),
        "pixelreplace2": np.array([0.0]),
        "worldmap0": np.array([0, 1], dtype=np.int32),
        "worldmap1": np.array([2], dtype=np.int32),
        "worldmap2": np.array([3], dtype=np.int32),
        "worldreplace0": np.array([0.0, 0.0]),
        "worldreplace1": np.array([1.0]),
        "worldreplace2": np.array([crval]),
    }
    return coords
