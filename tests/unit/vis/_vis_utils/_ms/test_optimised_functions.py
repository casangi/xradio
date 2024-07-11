import pytest
import numpy as np
import xradio._utils.array as opt


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
            np.array([1, 1, 1, 2, 4, 3]),
            np.array([1, 2, 3, 4]),
        ),
        (
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([1]),
        ),
    ],
)
def test_unique_1d(input_array, expected_unique_array):
    # Act
    unique_array = opt.unique_1d(input_array)

    # Assert
    np.testing.assert_array_almost_equal(unique_array, expected_unique_array)


def test_pairing_function():
    # Arrange
    input_array = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [11, 1],
            [1, 11],
            [129, 82],
            [82, 129],
            [0, 2047],
            [2047, 0],
            [2047, 2047],
        ]
    )
    expected_paired_array = np.array(
        [
            0,
            1,
            1048576,
            1048577,
            11534337,
            1048587,
            135266386,
            85983361,
            2047,
            2146435072,
            2146437119,
        ]
    )

    # Act
    paired_array = opt.pairing_function(input_array)

    # Assert
    np.testing.assert_array_almost_equal(paired_array, expected_paired_array)


def test_inverse_pairing_function():
    # Arrange
    paired_array = np.array(
        [
            0,
            1,
            1048576,
            1048577,
            11534337,
            1048587,
            135266386,
            85983361,
            2047,
            2146435072,
            2146437119,
        ]
    )
    expected_array = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [11, 1],
            [1, 11],
            [129, 82],
            [82, 129],
            [0, 2047],
            [2047, 0],
            [2047, 2047],
        ]
    )

    # Act
    array = opt.inverse_pairing_function(paired_array)

    # Assert
    np.testing.assert_array_almost_equal(array, expected_array)
