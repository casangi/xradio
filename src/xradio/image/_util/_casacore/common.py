from casacore import images
from contextlib import contextmanager
import numpy as np
from typing import Dict, Generator, List, Union


@contextmanager
def _open_image_ro(infile: str) -> Generator[images.image, None, None]:
    image = images.image(infile)
    try:
        yield image
    finally:
        # there is no obvious way to close a python-casacore image, so
        # just delete the object to clear it from the table cache
        del image


@contextmanager
def _create_new_image(
    outfile: str, shape: List[int], mask="", value="default"
) -> Generator[images.image, None, None]:
    # new image will be opened rw
    # the crux of the issue here seems to be that python has no single
    # precision floating point value, but we need single precision for
    # most images. The image constructor gets that right if values isn't
    # suppliked. Hence two calls with values not present and values present
    if value == "default":
        image = images.image(outfile, maskname=mask, shape=shape)
    else:
        image = images.image(outfile, maskname=mask, shape=shape, values=value)
    try:
        yield image
    finally:
        del image


_active_mask = "active_mask"
# _native_types = ["FREQ", "VRAD", "VOPT", "BETA", "WAVE", "AWAV"]
_object_name = "object_name"
_pointing_center = "pointing_center"
