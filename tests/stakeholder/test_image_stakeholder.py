import pathlib
import pytest
import toolviper

import numpy as np

from toolviper.utils.data import download

# relative_tolerance = 10 ** (-12)
relative_tolerance = 10 ** (-6)


def test_image(tmp_path: pathlib.Path):
    """Test image loading and writing with proper cleanup."""
    from xradio.image import load_image, open_image, write_image

    # Download test data
    image_name = "demo_simulated.im"
    toolviper.utils.data.download(file=image_name, folder=str(tmp_path))

    image_path = tmp_path / image_name
    zarr_output = tmp_path / "test_image.zarr"

    # Load images
    lazy_img_xds = open_image(str(image_path))

    img_xds = load_image(
        store=str(image_path),
        do_sky_coords=True,
    )

    print(img_xds)

    # Compute sums
    sum_result = np.nansum(np.abs(img_xds.SKY))
    sum_lazy = np.nansum(np.abs(lazy_img_xds.SKY))

    # Write output
    write_image(img_xds, str(zarr_output), out_format="zarr", overwrite=True)

    # Assertion
    assert np.isclose(
        sum_result, sum_lazy, rtol=relative_tolerance
    ), "read_image and load_image SKY sums differ."

    # Cleanup happens automatically with pytest's tmp_path fixture


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
