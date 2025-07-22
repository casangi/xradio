import importlib.resources
import numpy as np
import os
import pathlib
import pytest
import time

from toolviper.utils.data import download
from toolviper.utils.logger import setup_logger
from xradio.schema.check import check_dataset

# relative_tolerance = 10 ** (-12)
relative_tolerance = 10 ** (-6)


def test_image():
    from xradio.image import load_image, read_image, write_image

    image_name = "demo_simulated.im"
    download(file=image_name, folder="data")

    image_name = pathlib.Path.cwd().joinpath("data").joinpath("demo_simulated.im")
    lazy_img_xds = read_image(image_name)

    img_xds = load_image(
        infile= image_name,
        do_sky_coords=True,
    )

    sum = np.nansum(np.abs(img_xds.SKY))
    sum_lazy = np.nansum(np.abs(lazy_img_xds.SKY))

    write_image(img_xds, "test_image.zarr", out_format="zarr", overwrite=True)

    assert np.isclose(
        sum, sum_lazy, rtol=relative_tolerance
    ), "read_image and load_image SKY sums differ."

    os.system("rm -rf " + str(image_name))  # Remove image.


if __name__ == "__main__":
    a = 42
    from pathlib import Path

    test_image()
