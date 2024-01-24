import numpy as np


_np_types = {
    "complex128": np.complex128,
    "complex64": np.complex64,
    "float64": np.float64,
    "float16": np.float16,
    "float32": np.float32,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
}

if hasattr(np, "complex256"):
    _np_types["complex256"] = np.complex256
if hasattr(np, "float128"):
    _np_types["float128"] = np.float128

_top_level_sub_xds = "_attrs_xds"
