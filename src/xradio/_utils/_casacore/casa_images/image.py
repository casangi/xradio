"""A ``casacore.images.image``-compatible image class backed by casacoretables.

A CASA "paged image" is just a casacore table: the pixels live in a single
array column (``map``) and the metadata (coordinate system, image info,
brightness unit, masks) live in table keywords. ``casacoretables`` gives us the
full table layer, so this module reimplements the small slice of the
python-casacore ``image`` interface that xradio actually uses -- reading
metadata and shape, and creating/writing a paged image -- entirely on top of
tables. No casacore C++ images library (and therefore no casatools /
python-casacore) is required.

Only the methods exercised by xradio's CASA-image reader/writer are
implemented; this is deliberately not a full reimplementation of
``casacore.images.image``.
"""

import ast
import os
import shutil
from typing import List

import numpy as np
import numpy.ma as nma

from casacoretables import tables
from casacoretables.tables import maketabdesc, makearrcoldesc, makescacoldesc

from .coordinates import coordinatesystem
from ._coords_postprocess import postprocess_coords
from ._default_coords import build_default_coords

# Valid casacore ImageInfo enum values (see ImageInfo::ImageTypes). xradio relies
# on image.info()["imageinfo"]["imagetype"] always being one of these, mirroring
# python-casacore which validates the on-disk value against the enum.
_VALID_IMAGE_TYPES = (
    "Undefined",
    "Intensity",
    "Beam",
    "ColumnDensity",
    "DepolarizationRatio",
    "KineticTemperature",
    "MagneticField",
    "OpticalDepth",
    "RotationMeasure",
    "RotationalTemperature",
    "SpectralIndex",
    "Velocity",
    "VelocityDispersion",
)

# The column that holds the pixels in a CASA paged image.
_PIXEL_COLUMN = "map"


def _validate_image_type(value: str) -> str:
    """Normalize an image-type string to a valid casacore ImageInfo enum value.

    Validation is case-insensitive; the canonical capitalization is returned.
    Anything not in the enum (e.g. ``"sky"``) maps to ``"Intensity"``, matching
    python-casacore's ``image.imageinfo()`` behavior.
    """
    value_lower = str(value).lower()
    for valid_type in _VALID_IMAGE_TYPES:
        if valid_type.lower() == value_lower:
            return valid_type
    return "Intensity"


def _casa_vtype_and_dtype(values):
    """Map a fill ``values`` scalar (or None) to (casacore valueType, numpy dtype).

    ``None`` -> single-precision float, matching casacore's default image type.
    """
    if values is None:
        return "float", np.float32
    arr = np.asarray(values)
    dt = arr.dtype
    if dt == np.bool_:
        return "bool", np.bool_
    if np.issubdtype(dt, np.complexfloating):
        if dt == np.complex128 and isinstance(values, (np.complex128,)):
            return "dcomplex", np.complex128
        return "complex", np.complex64
    if dt == np.float32:
        return "float", np.float32
    if np.issubdtype(dt, np.floating):
        return "double", np.float64
    # integers etc.: store as double, as casacore would for a generic value
    return "double", np.float64


def _create_paged_table(path: str, np_shape, vtype: str, fill, dtype) -> None:
    """Create a one-row paged-image table with a ``map`` array column."""
    np_shape = tuple(int(s) for s in np_shape)
    ndim = len(np_shape)
    casa_shape = list(np_shape[::-1])
    zero = np.zeros((), dtype=dtype).item()
    coldesc = makearrcoldesc(_PIXEL_COLUMN, zero, ndim=ndim, valuetype=vtype)
    tabdesc = maketabdesc([coldesc])
    dminfo = {
        "*1": {
            "TYPE": "TiledShapeStMan",
            "NAME": _PIXEL_COLUMN,
            "SPEC": {"DEFAULTTILESHAPE": np.array(casa_shape, dtype=np.int32)},
            "COLUMNS": [_PIXEL_COLUMN],
        }
    }
    shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(path):
        os.remove(path)
    tb = tables.table(path, tabdesc, nrow=1, dminfo=dminfo, ack=False)
    # Define the cell shape (and fill value) with a full-cell write.
    tb.putcell(_PIXEL_COLUMN, 0, np.full(np_shape, fill, dtype=dtype))
    tb.putinfo({"type": "Image", "subType": "", "readme": ""})
    tb.close()


# Record used for the ``masks`` keyword entry of a CASA internal mask
# (LCPagedMask). Only the keys of the ``masks`` record (the mask names) are read
# back by xradio, but we write a faithful record for CASA interoperability.
def _mask_record(mask_path: str, casa_shape) -> dict:
    casa_shape = np.array(list(casa_shape), dtype=np.float64)
    return {
        "box": {
            "blc": np.ones(len(casa_shape), dtype=np.float64),
            "comment": "",
            "isRegion": 1,
            "name": "LCBox",
            "oneRel": True,
            "shape": casa_shape,
            "trc": casa_shape,
        },
        "comment": "",
        "isRegion": 1,
        "name": "LCPagedMask",
        "mask": f"Table: {mask_path}",
    }


class image:
    """python-casacore ``image``-compatible object backed by casacoretables.

    Supports the two construction modes xradio uses:

    * ``image(name)`` -- open an existing CASA paged image read-only.
    * ``image(name, maskname=..., shape=..., values=...)`` -- create a new
      paged image of the given shape (numpy order). ``values`` (a scalar) sets
      the pixel data type and fill value; ``None`` means single-precision float.
      A boolean mask sub-image ``maskname`` is created and registered if given.
    """

    def __init__(
        self,
        imagename,
        axis=0,
        maskname="",
        images=(),
        values=None,
        coordsys=None,
        overwrite=True,
        ashdf5=False,
        mask=(),
        shape=None,
        tileshape=(),
    ):
        if not isinstance(imagename, str):
            raise ValueError("first argument must be the image name (str)")
        self._path = os.path.expanduser(imagename)
        self._maskname = maskname or ""
        self._table = None  # open read-only handle for the read path

        if shape is None:
            # Open an existing image.
            self._table = tables.table(
                self._path,
                readonly=True,
                lockoptions={"option": "usernoread"},
                ack=False,
            )
        else:
            self._create(shape, values, coordsys)

    # ------------------------------------------------------------------ #
    # Creation
    # ------------------------------------------------------------------ #
    def _create(self, shape, values, coordsys):
        np_shape = tuple(int(s) for s in shape)
        vtype, dtype = _casa_vtype_and_dtype(values)
        fill = 0.0 if values is None else np.asarray(values).item()
        _create_paged_table(self._path, np_shape, vtype, fill, dtype)

        coords_rec = (
            coordsys.dict() if coordsys is not None else build_default_coords(np_shape)
        )
        with tables.table(
            self._path,
            readonly=False,
            lockoptions={"option": "permanentwait"},
            ack=False,
        ) as tb:
            tb.putkeyword("coords", coords_rec)
            tb.putkeyword("imageinfo", {"imagetype": "Intensity", "objectname": ""})

        self._create_logtable()
        if self._maskname:
            self._create_mask(np_shape)

    def _create_logtable(self):
        """Create the empty history ``logtable`` that every CASA image carries."""
        log_path = os.sep.join([self._path, "logtable"])
        cols = [
            makescacoldesc("TIME", 0.0, valuetype="double"),
            makescacoldesc("PRIORITY", "", valuetype="string"),
            makescacoldesc("MESSAGE", "", valuetype="string"),
            makescacoldesc("LOCATION", "", valuetype="string"),
            makescacoldesc("OBJECT_ID", "", valuetype="string"),
        ]
        tabdesc = maketabdesc(cols)
        tb = tables.table(log_path, tabdesc, nrow=0, ack=False)
        tb.putinfo(
            {
                "type": "Log message",
                "subType": "",
                "readme": "Repository for software-generated logging messages\n",
            }
        )
        tb.close()
        with tables.table(
            self._path,
            readonly=False,
            lockoptions={"option": "permanentwait"},
            ack=False,
        ) as main:
            main.putkeyword("logtable", f"Table: {log_path}")

    def _create_mask(self, np_shape):
        mask_path = os.sep.join([self._path, self._maskname])
        # casacore convention: True == good. A freshly created mask is all good.
        _create_paged_table(mask_path, np_shape, "bool", True, np.bool_)
        casa_shape = list(np_shape[::-1])
        with tables.table(
            self._path,
            readonly=False,
            lockoptions={"option": "permanentwait"},
            ack=False,
        ) as tb:
            tb.putkeyword(
                "masks", {self._maskname: _mask_record(mask_path, casa_shape)}
            )
            tb.putkeyword("Image_defaultmask", self._maskname)

    # ------------------------------------------------------------------ #
    # Metadata accessors
    # ------------------------------------------------------------------ #
    def _ro_table(self):
        if self._table is not None:
            return self._table
        return tables.table(
            self._path,
            readonly=True,
            lockoptions={"option": "usernoread"},
            ack=False,
        )

    def name(self, strippath=False):
        """Get the image name (path)."""
        if strippath:
            return os.path.basename(self._path)
        return self._path

    def shape(self) -> List[int]:
        """Get the image shape in numpy (C) order."""
        tb = self._ro_table()
        shp = ast.literal_eval(tb.getcolshapestring(_PIXEL_COLUMN)[0])
        return list(map(int, shp))

    def ndim(self) -> int:
        return len(self.shape())

    def datatype(self) -> str:
        """Get the pixel data type ('float', 'double', 'complex', 'dcomplex')."""
        tb = self._ro_table()
        return tb.coldatatype(_PIXEL_COLUMN)

    def coordinates(self) -> coordinatesystem:
        """Get the image coordinate system."""
        tb = self._ro_table()
        return coordinatesystem(postprocess_coords(tb.getkeyword("coords")))

    def imageinfo(self) -> dict:
        """Get the standard image info, with a validated imagetype."""
        tb = self._ro_table()
        if "imageinfo" in tb.keywordnames():
            info = tb.getkeyword("imageinfo")
        else:
            info = {"imagetype": "Intensity", "objectname": ""}
        info["imagetype"] = _validate_image_type(info.get("imagetype", "Intensity"))
        return info

    def miscinfo(self) -> dict:
        tb = self._ro_table()
        if "miscinfo" in tb.keywordnames():
            return tb.getkeyword("miscinfo")
        return {}

    def unit(self) -> str:
        """Get the brightness (pixel) unit."""
        tb = self._ro_table()
        if "units" in tb.keywordnames():
            return tb.getkeyword("units")
        return ""

    def info(self) -> dict:
        """Get coordinates, image info, misc info and unit (python-casacore form)."""
        tb = self._ro_table()
        coords = postprocess_coords(tb.getkeyword("coords"))
        return {
            "coordinates": coords,
            "imageinfo": self.imageinfo(),
            "miscinfo": self.miscinfo(),
            "unit": self.unit(),
        }

    # ------------------------------------------------------------------ #
    # Pixel / mask data
    # ------------------------------------------------------------------ #
    def getdata(self, blc=(), trc=(), inc=()) -> np.ndarray:
        """Get the full image pixel array (numpy order)."""
        tb = self._ro_table()
        return tb.getcell(tb.colnames()[0], 0)

    def _default_mask_name(self) -> str:
        """Name of the image's active (default) mask, or '' if it has none."""
        if self._maskname:
            return self._maskname
        tb = self._ro_table()
        if "Image_defaultmask" in tb.keywordnames():
            return tb.getkeyword("Image_defaultmask")
        return ""

    def getmask(self, blc=(), trc=(), inc=()) -> np.ndarray:
        """Get the image mask as a numpy array (True == bad, numpy convention).

        If the image has no mask, an all-False array is returned.
        """
        maskname = self._default_mask_name()
        if not maskname:
            return np.zeros(self.shape(), dtype=bool)
        with tables.table(
            os.sep.join([self._path, maskname]),
            readonly=True,
            lockoptions={"option": "usernoread"},
            ack=False,
        ) as tb:
            casa_mask = tb.getcell(tb.colnames()[0], 0)
        # casacore True == good -> numpy True == bad
        return np.logical_not(casa_mask)

    def tofits(
        self,
        filename,
        overwrite=True,
        velocity=True,
        optical=True,
        bitpix=-32,
        minpix=1,
        maxpix=-1,
    ) -> None:
        """Write this image to a FITS file (primary HDU + BEAMS table if multibeam)."""
        from ._tofits import to_fits

        to_fits(self, filename, overwrite=overwrite)

    def put(self, value, blc=(), trc=(), inc=()) -> None:
        """Write image data, and the mask if ``value`` is a numpy masked array."""
        if isinstance(value, nma.MaskedArray):
            self._put_pixels(self._path, np.asarray(value.data))
            if self._maskname:
                # casacore True == good -> negate numpy mask (True == bad)
                good = np.logical_not(nma.getmaskarray(value))
                self._put_pixels(os.sep.join([self._path, self._maskname]), good)
        else:
            self._put_pixels(self._path, np.asarray(value))

    def _put_pixels(self, path: str, arr: np.ndarray) -> None:
        with tables.table(
            path,
            readonly=False,
            lockoptions={"option": "permanentwait"},
            ack=False,
        ) as tb:
            tb.putcell(_PIXEL_COLUMN, 0, arr)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def unlock(self) -> None:
        """Release the read handle / flush. No-op for write paths."""
        if self._table is not None:
            try:
                self._table.unlock()
            except Exception:
                pass

    def close(self) -> None:
        if self._table is not None:
            try:
                self._table.close()
            except Exception:
                pass
            self._table = None

    def __del__(self):
        self.close()
