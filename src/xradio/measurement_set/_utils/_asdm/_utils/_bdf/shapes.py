"""
Calculations with the array shapes of auto/cross data arrays
"""


def add_cross_and_auto_flag_shapes(
    guessed_shape: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    guessed_shape_cross = guessed_shape["cross"]
    guessed_shape_auto = guessed_shape["auto"]
    if guessed_shape_cross:
        # second dim is the "BAL ANT"
        shape = (
            guessed_shape_cross[0],
            guessed_shape_cross[1] + guessed_shape_auto[1],
            *guessed_shape_cross[2:],
        )
    else:
        # The axes of flags would be for example "TIM ANT"
        # or something with ANT but not BAL
        shape = guessed_shape_auto

    return shape


def full_shape_to_output_filled_flags_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    # equivalent to the squeezing that would happen when selecting
    # one baseband / one SPW with int indices.
    return shape[0:2] + shape[-1:]
