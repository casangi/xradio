# methods in cngi_io.image.image should be used going forward,
# methods in cngi_io.image.cngi_image_io are deprecated
import sys
import warnings

from xradio.image.image import (
    load_image,
    make_empty_aperture_image,
    make_empty_lmuv_image,
    make_empty_sky_image,
    open_image,
    write_image,
)
from xradio.image.image_xds import _IMAGE_XDS_DEPRECATION, ImageXdsAccessor

# Importing the schema module registers the "image_dataset" type with the
# schema checker (xradio.schema.check.check_datatree). The dataset schema
# class itself is xradio.image.schema.ImageXds; the class of the xr_img
# accessor is ImageXdsAccessor (formerly ImageXds, a deprecated alias now).
from xradio.image.schema import check_image

__all__ = [
    "ImageXdsAccessor",
    "check_image",
    "load_image",
    "make_empty_aperture_image",
    "make_empty_lmuv_image",
    "make_empty_sky_image",
    "open_image",
    "write_image",
]


def __getattr__(name: str):
    # ImageXds: deprecated alias of ImageXdsAccessor
    if name == "ImageXds":
        # "from xradio.image import ImageXds" looks the name up twice: first
        # importlib's hasattr probe (_handle_fromlist), then the real lookup.
        # Warn on the real one only, so each use warns once.
        caller = sys._getframe().f_back
        if caller is None or caller.f_code.co_name != "_handle_fromlist":
            warnings.warn(_IMAGE_XDS_DEPRECATION, DeprecationWarning, stacklevel=2)
        return ImageXdsAccessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
