"""Primary beam models.

Schema of the Zernike polynomial coefficient dataset (``zpc_xds``), which
models the complex aperture illumination of an antenna, see
:py:mod:`xradio.primary_beam.schema`, and a reader for the coefficient tables
of Sekhar et al. (2022), :py:func:`read_zpc_csv`.
"""

from xradio.primary_beam.read_zpc_csv import read_zpc_csv
from xradio.primary_beam.schema import (
    ZPC_SCHEMA_VERSION,
    ZernikePolynomialCoefficientsXds,
    check_zpc,
)

__all__ = [
    "ZPC_SCHEMA_VERSION",
    "ZernikePolynomialCoefficientsXds",
    "check_zpc",
    "read_zpc_csv",
]
