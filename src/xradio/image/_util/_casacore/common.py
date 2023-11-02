from casacore import images
from contextlib import contextmanager
from typing import Dict, Generator, List

__active_mask:str = 'active_mask'
__native_types:List[str] = ['FREQ', 'VRAD', 'VOPT', 'BETA', 'WAVE', 'AWAV']
__object_name:str = 'object_name'
__pointing_center:str = 'pointing_center'


@contextmanager
def __open_image_ro(infile:str) -> Generator[images.image, None, None]:
    image = images.image(infile)
    try:
        yield image
    finally:
        # there is no obvious way to close a python-casacore image, so
        # just delete the object to clear it from the table cache
        del image


@contextmanager
def __open_new_image(outfile:str, shape:List[int]) -> Generator[images.image, None, None]:
    # new image will be opened rw
    image = images.image(outfile, shape=shape)
    try:
        yield image
    finally:
        del image
