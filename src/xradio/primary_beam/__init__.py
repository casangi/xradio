"""Primary beam models.

Schema of the Zernike polynomial coefficient dataset (``zpc_xds``), which
models the complex aperture illumination of an antenna, see
:py:mod:`xradio.primary_beam.schema`.
"""

from xradio.primary_beam.schema import (
    ZPC_SCHEMA_VERSION,
    ZernikePolynomialCoefficientsXds,
    check_zpc,
)

__all__ = ["ZPC_SCHEMA_VERSION", "ZernikePolynomialCoefficientsXds", "check_zpc"]
