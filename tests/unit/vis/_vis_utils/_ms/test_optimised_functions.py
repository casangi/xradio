import pytest
import numpy as np
import xradio.vis._vis_utils._ms.optimised_functions as opt


@pytest.mark.parametrize(
    "input_array, expected_unique_array",
    [
        (
            np.array(
                [
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                ]
            ),
            np.array([1, 2, 3]),
        ),
        (
            np.array(
                [
                    1,
                    1,
                    1,
                    2,
                    4,
                    3
                ]
            ),
            np.array([1, 2, 3, 4]),
        ),
        (
            np.array(
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1
                ]
            ),
            np.array([1]),
        ),
    ],
)
def test_unique_1d(input_array, expected_unique_array):
    # Act
    unique_array = opt.unique_1d(input_array)

    # Assert
    np.testing.assert_array_almost_equal(unique_array, expected_unique_array)
