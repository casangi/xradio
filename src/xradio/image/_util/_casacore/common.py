from casacore import images
from contextlib import contextmanager
from typing import Dict, Generator, List


@contextmanager
def _open_image_ro(infile:str) -> Generator[images.image, None, None]:
    image = images.image(infile)
    try:
        yield image
    finally:
        # there is no obvious way to close a python-casacore image, so
        # just delete the object to clear it from the table cache
        del image


@contextmanager
def _open_new_image(outfile:str, shape:List[int]) -> Generator[images.image, None, None]:
    # new image will be opened rw
    image = images.image(outfile, shape=shape)
    try:
        yield image
    finally:
        del image


_active_mask = "active_mask"
_native_types = ["FREQ", "VRAD", "VOPT", "BETA", "WAVE", "AWAV"]
_object_name = "object_name"
_pointing_center = "pointing_center"
