import importlib.resources
import numpy as np
import os
import pathlib
import pytest
import time

from graphviper.utils.data import download
from graphviper.utils.logger import setup_logger
from xradio.vis import (
    read_processing_set,
    load_processing_set,
    convert_msv2_to_processing_set,
    VisibilityXds,
)
from xradio.schema.check import check_dataset

# relative_tolerance = 10 ** (-12)
relative_tolerance = 10 ** (-6)


def test_image():
    image_name = "demo_simulated.im"
    download(image_name)
    from xradio.image import load_image, read_image, write_image

    lazy_img_xds = read_image(image_name)

    img_xds = load_image(
        image_name,
        do_sky_coords=True,
    )

    sum = np.nansum(np.abs(img_xds.SKY))
    sum_lazy = np.nansum(np.abs(lazy_img_xds.SKY))

    assert sum == sum_lazy, "read_image and load_image SKY sums differ."

    os.system("rm -rf " + str(image_name))  # Remove image.


if __name__ == "__main__":
    a = 42
    from pathlib import Path

    test_image()
