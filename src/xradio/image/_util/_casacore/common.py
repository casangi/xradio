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


"""
@contextmanager
def _open_image_rw(
    infile: str, mask: str, shape: tuple
) -> Generator[images.image, None, None]:
    image = images.image(infile, maskname=mask, shape=shape)
    try:
        yield image
    finally:
        del image
"""


@contextmanager
def _create_new_image(
    outfile: str, shape: List[int], mask="", value=np.float32(0.0)
) -> Generator[images.image, None, None]:
    # new image will be opened rw
    image = images.image(outfile, maskname=mask, shape=shape, values=value)
    try:
        yield image
    finally:
        del image


_active_mask = "active_mask"
# _native_types = ["FREQ", "VRAD", "VOPT", "BETA", "WAVE", "AWAV"]
_object_name = "object_name"
_pointing_center = "pointing_center"
