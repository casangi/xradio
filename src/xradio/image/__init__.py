# methods in cngi_io.image.image should be used going forward,
# methods in cngi_io.image.cngi_image_io are deprecated
from xradio.image.image import (
    load_image,
    make_empty_aperture_image,
    make_empty_lmuv_image,
    make_empty_sky_image,
    open_image,
    write_image,
)
from xradio.image.image_xds import ImageXds

# Importing the schema module registers the "image_dataset" type with the
# schema checker (xradio.schema.check.check_datatree). The dataset schema
# class itself is xradio.image.schema.ImageXds (not to be confused with the
# xr_img accessor class ImageXds exported here).
from xradio.image.schema import check_image

__all__ = [
    "ImageXds",
    "check_image",
    "load_image",
    "make_empty_aperture_image",
    "make_empty_lmuv_image",
    "make_empty_sky_image",
    "open_image",
    "write_image",
]
