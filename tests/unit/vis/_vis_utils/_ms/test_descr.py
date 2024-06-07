from contextlib import nullcontext as does_not_raise
import pandas as pd
import pytest

@pytest.mark.parametrize("name, mode, expected_raises", [
    ("inexistent_not_there.not_ms", "summary",
     pytest.raises(ValueError, match="invalid input")),
    ("test_ms_minimal_required.ms", "bogus",
     pytest.raises(ValueError, match="invalid mode")),
    ("test_ms_minimal_required.ms", "flat", does_not_raise()),
    ("test_ms_minimal_required.ms",  "expanded",
     pytest.raises(ValueError, match="update sequence element")),

])
def test_describe_ms_raises(name, mode, expected_raises):
    from xradio.vis._vis_utils._ms.descr import describe_ms
    with expected_raises:
        describe_ms(name, mode=mode)


@pytest.mark.parametrize("mode, expected_res", [
    ("summary", (pd.DataFrame, 4)),
    ("flat", (dict, 4)),
])
def test_describe_ms(ms_minimal_required, mode, expected_res):
    from xradio.vis._vis_utils._ms.descr import describe_ms

    res = describe_ms(ms_minimal_required.fname, mode)
    assert type(res) == expected_res[0]
    assert len(res) == expected_res[1]
