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
    # if os.environ["USER"] == "runner":
    #     casa_data_dir = (importlib.resources.files("casadata") / "__data__").as_posix()
    #     rc_file = open(os.path.expanduser("~/.casarc"), "a+")  # append mode
    #     rc_file.write("\nmeasures.directory: " + casa_data_dir)
    #     rc_file.close()

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

    assert np.isclose(sum, sum_lazy,rtol=relative_tolerance), "read_image and load_image SKY sums differ."

    os.system("rm -rf " + str(image_name))  # Remove image.


if __name__ == "__main__":
    a = 42
    from pathlib import Path

    test_image()
