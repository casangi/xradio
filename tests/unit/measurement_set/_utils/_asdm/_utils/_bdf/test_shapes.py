import pytest

example_guessed_shape_auto_only = {
    "auto": (5, 9, 1, 64, 2),
    "cross": None,
}
example_guessed_shape = {
    "cross": (3, 36, 1, 64, 2, 2),
    "auto": (5, 9, 1, 64, 2),
}


@pytest.mark.parametrize(
    "input_guessed_shape, expected_out_shape",
    [
        (example_guessed_shape, (3, 45, 1, 64, 2, 2)),
        (example_guessed_shape_auto_only, (5, 9, 1, 64, 2)),
    ],
)
def test_add_cross_and_auto_flag_shapes_with_cross(
    input_guessed_shape, expected_out_shape
):
    from xradio.measurement_set._utils._asdm._utils._bdf.shapes import (
        add_cross_and_auto_flag_shapes,
    )

    added = add_cross_and_auto_flag_shapes(input_guessed_shape)
    assert added == expected_out_shape


@pytest.mark.parametrize(
    "input_shape, expected_out_shape",
    [
        (example_guessed_shape["auto"], (5, 9, 2)),
        (example_guessed_shape["cross"], (3, 36, 2)),
    ],
)
def test_full_shape_to_output_filled_flags_shape(input_shape, expected_out_shape):
    from xradio.measurement_set._utils._asdm._utils._bdf.shapes import (
        full_shape_to_output_filled_flags_shape,
    )

    out_shape = full_shape_to_output_filled_flags_shape(input_shape)
    assert out_shape == expected_out_shape
