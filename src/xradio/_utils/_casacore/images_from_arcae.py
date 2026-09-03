"""
python-casacore ``casacore.images``-compatible module on top of arcae.

arcae has no image API, but CASA images are stored as casacore tables: the
pixels live in a single-cell ``TiledCellStMan`` column (``map``) and all
metadata (coordinate system, image info, units, masks) lives in table
keywords.  This module therefore implements the subset of
``casacore.images.image`` and ``casacore.images.coordinates`` that XRADIO
uses purely with table access through
:mod:`xradio._utils._casacore.casacore_from_arcae`:

* metadata comes from the table keywords (``coords``, ``imageinfo``,
  ``units``, ``miscinfo``);
* the image shape comes from a TaQL ``shape()`` expression (arcae cannot
  read ``TiledCellStMan`` cells directly);
* new images are created with TaQL ``CREATE TABLE`` (tiled data column,
  optional default mask subtable and an empty CASA log table).

The module doubles as a stand-in for both the ``casacore.images`` and the
``casacore.images.coordinates`` namespaces (it defines ``image`` as well as
the coordinate wrapper classes).
"""

import os
import re

import numpy as np

from xradio._utils._casacore import casacore_from_arcae as _tables

# ---------------------------------------------------------------------------
# coordinate system wrappers (API-compatible with python-casacore's
# casacore.images.coordinates, operating on the image "coords" record)
# ---------------------------------------------------------------------------

_COORDINATE_TYPES = ("direction", "spectral", "linear", "stokes", "tabular")


class coordinate:
    """A single coordinate of a coordinate system.

    All list-valued quantities are returned reversed (C order), while the
    underlying record stores them in Fortran order — this matches
    python-casacore behavior.
    """

    def __init__(self, rec):
        self._coord = rec

    def dict(self):
        return self._coord

    def get_referencepixel(self):
        return self._coord.get("crpix", [])[::-1]

    def get_referencevalue(self):
        return self._coord.get("crval", [])[::-1]

    def get_increment(self):
        return self._coord.get("cdelt", [])[::-1]

    def get_unit(self):
        return self._coord.get("units", [])[::-1]

    def get_axes(self):
        return self._coord.get("axes", [])[::-1]


class directioncoordinate(coordinate):
    def get_projection(self):
        return self._coord.get("projection", None)

    def get_frame(self):
        return self._coord.get("system", None)


class spectralcoordinate(coordinate):
    def get_unit(self):
        return self._coord.get("unit", None)

    def get_referencepixel(self):
        return self._coord["wcs"].get("crpix", None)

    def get_referencevalue(self):
        return self._coord["wcs"].get("crval", None)

    def get_increment(self):
        return self._coord["wcs"].get("cdelt", None)

    def get_axes(self):
        return self._coord.get("name", None)

    def get_restfrequency(self):
        return self._coord.get("restfreq", None)

    def get_frame(self):
        return self._coord.get("system", None)


class linearcoordinate(coordinate):
    pass


class stokescoordinate(coordinate):
    def get_stokes(self):
        return self._coord["stokes"]


class tabularcoordinate(coordinate):
    pass


_COORDINATE_CLASSES = {
    "direction": directioncoordinate,
    "spectral": spectralcoordinate,
    "linear": linearcoordinate,
    "stokes": stokescoordinate,
    "tabular": tabularcoordinate,
}


class coordinatesystem:
    """Wrapper for a CASA image coordinate-system record.

    The coordinate names are held in reverse (C) order with respect to the
    ``<type><i>`` numbering in the record, exactly like python-casacore.
    """

    def __init__(self, rec):
        self._csys = rec
        names = [""] * len(_COORDINATE_TYPES)
        n = 0
        for key in rec:
            for name in _COORDINATE_TYPES:
                if key.startswith(name):
                    suffix = key[len(name) :]
                    if suffix.isdigit():
                        names[int(suffix)] = name
                        n += 1
        self._names = names[:n][::-1]
        if not self._names:
            raise LookupError("Coordinate record doesn't contain valid coordinates")

    def dict(self):
        return self._csys

    def get_names(self):
        return self._names

    def get_coordinate(self, name):
        # reverse index back to Fortran order used by the record keys
        i = self._names[::-1].index(name)
        return _COORDINATE_CLASSES[name](self._csys[name + str(i)])

    __getitem__ = get_coordinate

    def __iter__(self):
        for name in self._names:
            yield self.get_coordinate(name)

    def get_obsdate(self):
        return self._csys.get("obsdate", None)

    def get_observer(self):
        return self._csys.get("observer", None)

    def get_telescope(self):
        return self._csys.get("telescope", None)

    def get_referencepixel(self):
        return [coord.get_referencepixel() for coord in self]

    def get_referencevalue(self):
        return [coord.get_referencevalue() for coord in self]

    def get_increment(self):
        return [coord.get_increment() for coord in self]

    def get_unit(self):
        return [coord.get_unit() for coord in self]

    def get_axes(self):
        return [coord.get_axes() for coord in self]


# ---------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------

_DTYPE_TO_VALUETYPE = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.complex64): "complex",
    np.dtype(np.complex128): "dcomplex",
}

# TaQL column data type codes for the pixel column
_VALUETYPE_TO_TAQL = {
    "float": "R4",
    "double": "R8",
    "complex": "C4",
    "dcomplex": "C8",
    "boolean": "B",
}

_LOG_COLUMNS = {
    "TIME": {
        "valueType": "double",
        "option": 0,
        "maxlen": 0,
        "comment": "MJD in seconds",
        "keywords": {
            "UNIT": "s",
            "MEASURE_TYPE": "EPOCH",
            "MEASURE_REFERENCE": "UTC",
        },
    },
    "PRIORITY": {"valueType": "string", "option": 0, "maxlen": 0, "comment": ""},
    "MESSAGE": {"valueType": "string", "option": 0, "maxlen": 0, "comment": ""},
    "LOCATION": {"valueType": "string", "option": 0, "maxlen": 0, "comment": ""},
    "OBJECT_ID": {"valueType": "string", "option": 0, "maxlen": 0, "comment": ""},
}


def _default_tile_shape(shape_f):
    """Choose a tile shape (Fortran order) for a new tiled image column."""
    tile = []
    for i, dim in enumerate(shape_f):
        # generous tiles on the two fastest-varying (sky) axes, single-plane
        # tiles on the remaining axes
        tile.append(int(min(dim, 256 if i < 2 else 1)))
    return tile


def _write_table_info(tablename, info_type, readme=""):
    """Write the table.info file of a CLOSED table.

    casacore rewrites table.info from its in-memory state when the table is
    closed, so this must happen after the last handle to the table is gone.
    """
    if readme and not readme.endswith("\n"):
        readme += "\n"
    with open(os.path.join(tablename, "table.info"), "w") as f:
        f.write(f"Type = {info_type}\nSubType = \n\n{readme}")


def _create_tiled_table(tablename, columnname, valuetype, shape_f, info_type):
    """Create a 1-row table holding a single tiled array column and return
    an open read/write handle to it."""
    taql_type = _VALUETYPE_TO_TAQL[valuetype]
    shape_spec = "[" + ",".join(str(int(s)) for s in shape_f) + "]"
    tile_spec = "[" + ",".join(str(s) for s in _default_tile_shape(shape_f)) + "]"
    created = _tables._at.Table.from_taql(
        f"CREATE TABLE {_tables._quote_path(tablename)} "
        f"({columnname} {taql_type} "
        f"[NDIM={len(shape_f)}, SHAPE={shape_spec}, COMMENT='version 4.0']) "
        f"LIMIT 1 "
        f"DMINFO [TYPE='TiledCellStMan', NAME='{columnname}', "
        f"COLUMNS=['{columnname}'], SPEC=[DEFAULTTILESHAPE={tile_spec}]]"
    )
    created.close()
    _write_table_info(tablename, info_type)
    return _tables.table(
        tablename,
        readonly=False,
        lockoptions={"option": "permanentwait"},
        ack=False,
    )


_HI_RESTFREQ = 1420405751.786

# casacore ImageInfo::ImageTypes names
_CASA_IMAGE_TYPES = frozenset(
    (
        "Undefined",
        "Intensity",
        "Beam",
        "Column Density",
        "Depolarization Ratio",
        "Kinetic Temperature",
        "Magnetic Field",
        "Optical Depth",
        "Rotation Measure",
        "Rotational Temperature",
        "Spectral Index",
        "Velocity",
    )
)

# casacore Stokes.h enumeration values for the FITS STOKES axis
_STOKES_CODES = {
    "I": 1,
    "Q": 2,
    "U": 3,
    "V": 4,
    "RR": 5,
    "RL": 6,
    "LR": 7,
    "LL": 8,
    "XX": 9,
    "XY": 10,
    "YX": 11,
    "YY": 12,
}

# casacore frequency frame -> FITS SPECSYS
_SPECSYS_MAP = {
    "REST": "SOURCE",
    "LSRK": "LSRK",
    "LSRD": "LSRD",
    "BARY": "BARYCENT",
    "GEO": "GEOCENTR",
    "TOPO": "TOPOCENT",
    "GALACTO": "GALACTOC",
    "LGROUP": "LOCALGRP",
    "CMB": "CMBDIPOL",
}


# casacore unit spellings that astropy does not parse directly
_CASA_UNIT_ALIASES = {"'": "arcmin", "''": "arcsec", '"': "arcsec"}


def _casa_unit(unit):
    unit = str(unit)
    return _CASA_UNIT_ALIASES.get(unit, unit)


def _to_deg(value, unit):
    from astropy import units as u

    return float((value * u.Unit(_casa_unit(unit))).to("deg").value)


def _beams_hdu(perplanebeams):
    """Build the CASA multi-beam BEAMS binary table HDU."""
    from astropy.io import fits

    nchan = int(perplanebeams["nChannels"])
    npol = int(perplanebeams["nStokes"])
    bmaj, bmin, bpa, chans, pols = [], [], [], [], []
    for pol in range(npol):
        for chan in range(nchan):
            beam = perplanebeams[f"*{pol * nchan + chan}"]
            bmaj.append(_to_deg(beam["major"]["value"], beam["major"]["unit"]))
            bmin.append(_to_deg(beam["minor"]["value"], beam["minor"]["unit"]))
            bpa.append(
                _to_deg(beam["positionangle"]["value"], beam["positionangle"]["unit"])
            )
            chans.append(chan)
            pols.append(pol)
    columns = [
        fits.Column(name="BMAJ", format="E", unit="deg", array=np.array(bmaj)),
        fits.Column(name="BMIN", format="E", unit="deg", array=np.array(bmin)),
        fits.Column(name="BPA", format="E", unit="deg", array=np.array(bpa)),
        fits.Column(name="CHAN", format="J", array=np.array(chans)),
        fits.Column(name="POL", format="J", array=np.array(pols)),
    ]
    hdu = fits.BinTableHDU.from_columns(columns, name="BEAMS")
    hdu.header["NCHAN"] = nchan
    hdu.header["NPOL"] = npol
    return hdu


def _default_coords_record(shape_f):
    """Synthesize a default coordinate-system record for a new 4-d image,
    like casacore's CoordinateUtil does for images created without an
    explicit coordinate system (axes: RA, Dec, Stokes, Frequency)."""
    if len(shape_f) != 4:
        raise ValueError(
            f"Only 4-d images can be created without a coordinate system, "
            f"got shape {shape_f[::-1]}"
        )
    nx, ny, nstokes, nchan = shape_f
    arcsec = np.deg2rad(1.0 / 3600.0)
    stokes_names = ["I", "Q", "U", "V"][: max(nstokes, 1)]
    return {
        "telescope": "UNKNOWN",
        "observer": "UNKNOWN",
        "obsdate": {
            "type": "epoch",
            "refer": "UTC",
            "m0": {"value": 51544.0, "unit": "d"},
        },
        "pointingcenter": {"value": [0.0, 0.0], "initial": True},
        "direction0": {
            "system": "J2000",
            "projection": "SIN",
            "projection_parameters": [0.0, 0.0],
            "crval": [0.0, 0.0],
            "crpix": [nx / 2.0, ny / 2.0],
            "cdelt": [-arcsec, arcsec],
            "pc": [[1.0, 0.0], [0.0, 1.0]],
            "axes": ["Right Ascension", "Declination"],
            "units": ["rad", "rad"],
            "conversionSystem": "J2000",
            "longpole": 180.0,
            "latpole": 0.0,
        },
        "worldmap0": np.array([0, 1], dtype=np.int32),
        "worldreplace0": [0.0, 0.0],
        "pixelmap0": np.array([0, 1], dtype=np.int32),
        "pixelreplace0": [0.0, 0.0],
        "stokes1": {
            "axes": ["Stokes"],
            "stokes": stokes_names,
            "crval": [1.0],
            "crpix": [0.0],
            "cdelt": [1.0],
            "pc": [[1.0]],
        },
        "worldmap1": np.array([2], dtype=np.int32),
        "worldreplace1": [1.0],
        "pixelmap1": np.array([2], dtype=np.int32),
        "pixelreplace1": [0.0],
        "spectral2": {
            "version": 2,
            "system": "LSRK",
            "restfreq": _HI_RESTFREQ,
            "restfreqs": [_HI_RESTFREQ],
            "velType": 0,
            "nativeType": 0,
            "velUnit": "km/s",
            "waveUnit": "mm",
            "formatUnit": "",
            "name": "Frequency",
            "unit": "Hz",
            "wcs": {
                "ctype": "FREQ",
                "crval": 1.415e9,
                "crpix": 0.0,
                "cdelt": 1e3,
                "pc": 1.0,
            },
        },
        "worldmap2": np.array([3], dtype=np.int32),
        "worldreplace2": [1.415e9],
        "pixelmap2": np.array([3], dtype=np.int32),
        "pixelreplace2": [0.0],
    }


def _fill_direction_defaults(coords):
    """Normalize direction records of a raw stored coordinate record.

    casacore restores coordinate systems through wcslib, which recomputes
    the native pole (longpole/latpole) from the reference direction; the
    values stored on disk may be missing or stale (e.g. 0.0). The same
    normalization is applied here with astropy's wcslib bindings. Private
    helper fields (leading underscore) that image writers may add are
    dropped, like casacore ignores them.
    """
    from astropy.wcs import WCS

    for key in list(coords):
        sub = coords[key]
        if not isinstance(sub, dict):
            continue
        if key[:-1] in ("direction", "spectral", "stokes", "linear", "tabular"):
            sub = {k: v for k, v in sub.items() if not k.startswith("_")}
            coords[key] = sub
        if key.startswith("direction"):
            from astropy import units as u

            crval = np.asarray(sub.get("crval", [0.0, 0.0]), dtype=float)
            units = [str(u0) for u0 in sub.get("units", ["rad", "rad"])]
            crval_deg = [
                float((v * u.Unit(_casa_unit(unit))).to("deg").value)
                for v, unit in zip(crval, units, strict=False)
            ]
            projection = sub.get("projection", "SIN")
            wcs = WCS(naxis=2)
            wcs.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
            wcs.wcs.crval = crval_deg
            wcs.wcs.crpix = [1.0, 1.0]
            try:
                wcs.wcs.set()
                sub["longpole"] = float(wcs.wcs.lonpole)
                sub["latpole"] = float(wcs.wcs.latpole)
            except Exception:
                sub.setdefault("longpole", 180.0)
                sub.setdefault("latpole", float(crval_deg[1]))
        elif key.startswith("spectral") and "conversion" not in sub:
            # casacore synthesizes a default reference-conversion layer when
            # restoring a spectral coordinate saved without one
            sub["conversion"] = {
                "direction": {
                    "type": "direction",
                    "refer": "J2000",
                    "m0": {"unit": "rad", "value": 0.0},
                    "m1": {"unit": "rad", "value": np.pi / 2},
                },
                "position": {
                    "type": "position",
                    "refer": "ITRF",
                    "m0": {"unit": "rad", "value": 0.0},
                    "m1": {"unit": "rad", "value": 0.0},
                    "m2": {"unit": "m", "value": 0.0},
                },
                "epoch": {
                    "type": "epoch",
                    "refer": "LAST",
                    "m0": {"unit": "d", "value": 0.0},
                },
                "system": sub.get("system", "LSRK"),
            }
    return coords


def _make_default_mask(imagepath, maskname, shape_f):
    """Create a mask subtable (all pixels good) and return its ``masks``
    keyword record."""
    mask_path = os.path.join(imagepath, maskname)
    mask_tab = _create_tiled_table(
        mask_path, "PagedArray", "boolean", shape_f, "Paged Array"
    )
    # a fresh default mask marks every pixel as good
    mask_tab._taql("UPDATE $1 SET PagedArray = T").close()
    mask_tab.close()
    return {
        "isRegion": 1,
        "name": "LCPagedMask",
        "comment": "",
        "mask": f"Table: {mask_path}",
        "box": {
            "isRegion": 1,
            "name": "LCBox",
            "comment": "",
            "oneRel": True,
            "blc": [1.0] * len(shape_f),
            "trc": [float(s) for s in shape_f],
            "shape": shape_f,
        },
    }


def _create_log_table(imagepath):
    logpath = os.path.join(imagepath, "logtable")
    log_tab = _tables.table(logpath, tabledesc=dict(_LOG_COLUMNS), nrow=0)
    log_tab.close()
    _write_table_info(
        logpath,
        "Log message",
        "Repository for software-generated logging messages",
    )


class image:
    """python-casacore-compatible CASA image reader/creator over tables."""

    def __init__(self, imagename, maskname="", shape=None, values=None):
        self._path = os.path.abspath(os.path.expanduser(str(imagename)))
        if shape is None:
            # open an existing image read-only
            self._tab = _tables.table(
                self._path,
                readonly=True,
                lockoptions={"option": "usernoread"},
                ack=False,
            )
            self._column = self._tab.colnames()[0]
        else:
            self._create(maskname, shape, values)

    # -- creation -----------------------------------------------------------

    def _create(self, maskname, shape, values):
        # shape is given in C (numpy) order; the table stores Fortran order
        shape_c = [int(s) for s in shape]
        shape_f = shape_c[::-1]
        if values is None:
            valuetype = "float"
        else:
            dtype = np.asarray(values).dtype
            if dtype not in _DTYPE_TO_VALUETYPE:
                dtype = np.dtype(np.float64)
            valuetype = _DTYPE_TO_VALUETYPE[dtype]
        self._column = "map"
        tab = _create_tiled_table(self._path, "map", valuetype, shape_f, "Image")
        keywords = {
            "units": "",
            "imageinfo": {"imagetype": "Intensity", "objectname": ""},
            "miscinfo": {},
            "coords": _default_coords_record(shape_f),
        }
        if maskname:
            keywords["masks"] = {
                maskname: _make_default_mask(self._path, maskname, shape_f)
            }
            keywords["Image_defaultmask"] = maskname
        tab.putkeywords(keywords)
        _create_log_table(self._path)
        tab.putkeyword("logtable", f"Table: {os.path.join(self._path, 'logtable')}")
        tab.close()
        if values is not None and np.ndim(values) == 0:
            value = np.asarray(values).item()
            if isinstance(value, complex):
                literal = f"{value.real!r}{value.imag:+}i"
            else:
                literal = repr(value)
            fill = _tables.table(
                self._path,
                readonly=False,
                lockoptions={"option": "permanentwait"},
                ack=False,
            )
            fill._taql(f"UPDATE $1 SET map = {literal}").close()
            fill.close()
        # reopen read-only like a freshly constructed python-casacore image
        self._tab = _tables.table(self._path, readonly=True, ack=False)

    # -- metadata -----------------------------------------------------------

    def name(self, strippath=False):
        return os.path.basename(self._path) if strippath else self._path

    def _keyword(self, name, default=None):
        keywords = self._tab._keywords()
        if name not in keywords:
            return default
        return _tables._from_json_keyword(keywords[name])

    def info(self):
        # like python-casacore/casacore ImageInfo::toRecord, imageinfo always
        # carries (default) imagetype and objectname entries
        imageinfo = {"imagetype": "Intensity", "objectname": ""}
        imageinfo.update(self._keyword("imageinfo", {}))
        if imageinfo["imagetype"] not in _CASA_IMAGE_TYPES:
            # casacore maps unrecognized image types to Intensity on restore
            imageinfo["imagetype"] = "Intensity"
        return {
            "coordinates": self._coords(),
            "imageinfo": imageinfo,
            "miscinfo": self._keyword("miscinfo", {}),
            "unit": self._keyword("units", ""),
        }

    def _coords(self):
        return _fill_direction_defaults(self._keyword("coords", {}))

    def imageinfo(self):
        return self.info()["imageinfo"]

    def coordinates(self):
        return coordinatesystem(self._coords())

    def unit(self):
        return self._keyword("units", "")

    def miscinfo(self):
        return self._keyword("miscinfo", {})

    def shape(self):
        # C (numpy) order, like python-casacore
        result = self._tab._taql(f"SELECT shape({self._column}) AS S FROM $1")
        try:
            shape_f = result.getcol("S")[0]
        finally:
            result.close()
        return [int(s) for s in shape_f[::-1]]

    def datatype(self):
        return self._tab.coldatatype(self._column)

    def ndim(self):
        return len(self.shape())

    def unlock(self):
        pass

    def close(self):
        self._tab.close()

    # -- pixel/mask writes --------------------------------------------------

    def _normalize_box(self, value, blc):
        shape = self.shape()
        value = np.asarray(value)
        if value.ndim != len(shape):
            raise ValueError(
                f"value has {value.ndim} dimensions, image has {len(shape)}"
            )
        blc = list(blc) if blc else []
        blc += [0] * (len(shape) - len(blc))
        trc = [b + s - 1 for b, s in zip(blc, value.shape, strict=True)]
        return value, blc, trc

    def _write_slice(self, tablename, columnname, value, blc, trc):
        tab = _tables.table(
            tablename,
            readonly=False,
            lockoptions={"option": "permanentwait"},
            ack=False,
        )
        try:
            tab.putcellslice(columnname, 0, value, blc, trc)
        finally:
            tab.close()

    def putdata(self, value, blc=(), trc=(), inc=()):
        value, blc, trc = self._normalize_box(value, blc)
        self._write_slice(self._path, self._column, value, blc, trc)

    def putmask(self, value, blc=(), trc=(), inc=()):
        # numpy convention: True = masked/bad; casacore: True = good
        value, blc, trc = self._normalize_box(value, blc)
        maskname = self._keyword("Image_defaultmask", "")
        if not maskname:
            if not value.any():
                # like python-casacore: nothing to do for an all-good mask
                # on an image without a mask
                return
            maskname = "mask0"
            shape_f = self.shape()[::-1]
            mask_record = _make_default_mask(self._path, maskname, shape_f)
            tab = _tables.table(
                self._path,
                readonly=False,
                lockoptions={"option": "permanentwait"},
                ack=False,
            )
            try:
                masks = tab.getkeyword("masks") if "masks" in tab.keywordnames() else {}
                masks[maskname] = mask_record
                tab.putkeywords({"masks": masks, "Image_defaultmask": maskname})
            finally:
                tab.close()
        mask_path = os.path.join(self._path, maskname)
        self._write_slice(mask_path, "PagedArray", ~value, blc, trc)

    def put(self, value, blc=(), trc=(), inc=()):
        if isinstance(value, np.ma.MaskedArray):
            self.putdata(value.data, blc, trc, inc)
            self.putmask(np.ma.getmaskarray(value), blc, trc, inc)
        else:
            self.putdata(value, blc, trc, inc)

    # -- pixel/mask reads ---------------------------------------------------

    def _read_slice(self, tablename, columnname, blc, trc):
        tab = _tables.table(
            tablename,
            readonly=True,
            lockoptions={"option": "usernoread"},
            ack=False,
        )
        try:
            return tab.getcellslice(columnname, 0, blc, trc)
        finally:
            tab.close()

    def _normalize_read_box(self, blc, trc):
        shape = self.shape()
        blc = list(blc) if blc else []
        blc += [0] * (len(shape) - len(blc))
        trc = list(trc) if trc else []
        trc += [s - 1 for s in shape[len(trc) :]]
        return blc, trc

    def getdata(self, blc=(), trc=(), inc=()):
        blc, trc = self._normalize_read_box(blc, trc)
        return self._read_slice(self._path, self._column, blc, trc)

    def getmask(self, blc=(), trc=(), inc=()):
        # numpy convention: True = masked/bad
        blc, trc = self._normalize_read_box(blc, trc)
        maskname = self._keyword("Image_defaultmask", "")
        if not maskname:
            shape = [t - b + 1 for b, t in zip(blc, trc, strict=True)]
            return np.zeros(shape, dtype=bool)
        mask_path = os.path.join(self._path, maskname)
        return ~self._read_slice(mask_path, "PagedArray", blc, trc)

    def get(self, blc=(), trc=(), inc=()):
        return np.ma.masked_array(
            self.getdata(blc, trc, inc), self.getmask(blc, trc, inc)
        )

    # -- FITS export --------------------------------------------------------

    def tofits(self, filename, overwrite=True, **kwargs):
        """Write the image to a FITS file (astropy-based).

        Supports the standard CASA 4-axis images (direction, Stokes,
        frequency) including per-plane beams (CASAMBM convention).
        """
        from astropy.io import fits
        from astropy.time import Time

        info = self.info()
        coords = info["coordinates"]
        imageinfo = info["imageinfo"]
        data = self.getdata().astype(np.float32)
        mask = self.getmask()
        if mask.any():
            data = np.where(mask, np.nan, data)

        header = fits.Header()
        ndim = data.ndim
        for i in range(1, ndim + 1):
            header[f"CTYPE{i}"] = "LINEAR"
            header[f"CRVAL{i}"] = 0.0
            header[f"CDELT{i}"] = 1.0
            header[f"CRPIX{i}"] = 1.0
            header[f"CUNIT{i}"] = ""

        for key, sub in coords.items():
            if not isinstance(sub, dict) or key[:-1] not in (
                "direction",
                "spectral",
                "stokes",
                "linear",
            ):
                continue
            # FITS axis numbers (1-based, Fortran order) for this coordinate
            # Extract numeric suffix (handles double-digit indices like direction10)
            coord_idx = re.search(r"\d+", key).group()
            axes = [int(a) + 1 for a in np.atleast_1d(coords[f"pixelmap{coord_idx}"])]
            if key.startswith("direction"):
                projection = sub.get("projection", "SIN")
                for j, (fits_base, casa_idx) in enumerate(
                    zip(
                        (f"RA---{projection}", f"DEC--{projection}"),
                        (0, 1),
                        strict=True,
                    )
                ):
                    ax = axes[j]
                    unit = str(sub["units"][casa_idx])
                    scale = np.rad2deg(1.0) if unit.startswith("rad") else 1.0
                    header[f"CTYPE{ax}"] = fits_base
                    header[f"CRVAL{ax}"] = float(sub["crval"][casa_idx]) * scale
                    header[f"CDELT{ax}"] = float(sub["cdelt"][casa_idx]) * scale
                    header[f"CRPIX{ax}"] = float(sub["crpix"][casa_idx]) + 1.0
                    header[f"CUNIT{ax}"] = "deg"
                system = sub.get("system", "ICRS")
                if system == "J2000":
                    header["RADESYS"] = "FK5"
                    header["EQUINOX"] = 2000.0
                elif system == "B1950":
                    header["RADESYS"] = "FK4"
                    header["EQUINOX"] = 1950.0
                else:
                    header["RADESYS"] = system.upper()
                header["LONPOLE"] = float(sub.get("longpole", 180.0))
                header["LATPOLE"] = float(sub.get("latpole", 0.0))
                pc = np.asarray(sub.get("pc", np.eye(2)))
                for r in (0, 1):
                    for c in (0, 1):
                        header[f"PC{axes[r]}_{axes[c]}"] = float(pc[r][c])
            elif key.startswith("stokes"):
                codes = [_STOKES_CODES[s] for s in sub["stokes"]]
                ax = axes[0]
                header[f"CTYPE{ax}"] = "STOKES"
                header[f"CRVAL{ax}"] = float(codes[0])
                header[f"CDELT{ax}"] = float(
                    codes[1] - codes[0] if len(codes) > 1 else 1
                )
                header[f"CRPIX{ax}"] = 1.0
                header[f"CUNIT{ax}"] = ""
            elif key.startswith("spectral"):
                wcs = sub["wcs"]
                ax = axes[0]
                header[f"CTYPE{ax}"] = "FREQ"
                header[f"CRVAL{ax}"] = float(wcs["crval"])
                header[f"CDELT{ax}"] = float(wcs["cdelt"])
                header[f"CRPIX{ax}"] = float(wcs["crpix"]) + 1.0
                header[f"CUNIT{ax}"] = str(sub.get("unit", "Hz"))
                header["RESTFRQ"] = float(sub.get("restfreq", 0.0))
                header["SPECSYS"] = _SPECSYS_MAP.get(
                    sub.get("system", "TOPO"), "TOPOCENT"
                )

        header["BUNIT"] = info.get("unit", "") or ""
        header["OBJECT"] = imageinfo.get("objectname", "")
        header["BTYPE"] = imageinfo.get("imagetype", "Intensity")
        header["TELESCOP"] = coords.get("telescope", "UNKNOWN")
        header["OBSERVER"] = coords.get("observer", "")
        obsdate = coords.get("obsdate")
        if obsdate:
            time = Time(
                float(obsdate["m0"]["value"]),
                format="mjd",
                scale=str(obsdate.get("refer", "UTC")).lower(),
            )
            header["DATE-OBS"] = time.isot
            header["TIMESYS"] = str(obsdate.get("refer", "UTC")).upper()

        extra_hdus = []
        if "perplanebeams" in imageinfo:
            header["CASAMBM"] = True
            extra_hdus.append(_beams_hdu(imageinfo["perplanebeams"]))
        elif "restoringbeam" in imageinfo:
            beam = imageinfo["restoringbeam"]
            for fits_key, casa_key in (
                ("BMAJ", "major"),
                ("BMIN", "minor"),
                ("BPA", "positionangle"),
            ):
                quantity = beam[casa_key]
                header[fits_key] = _to_deg(quantity["value"], quantity["unit"])
        hdus = [fits.PrimaryHDU(data=data, header=header)] + extra_hdus
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
