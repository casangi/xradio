"""A casacoretables-backed replacement for ``casacore.images``.

This subpackage provides just enough of the python-casacore ``casacore.images``
API for xradio's CASA-image reader/writer to work on top of ``casacoretables``
(which ships only casacore's table layer, not the images library):

* :class:`image` -- read metadata/shape of, and create/write, a CASA paged
  image, implemented entirely via tables.
* :mod:`coordinates` -- the pure-Python coordinate-system wrappers, vendored
  verbatim from python-casacore.

Import it the same way you would import ``casacore.images``::

    from xradio._utils._casacore.casa_images import coordinates, image
"""

from . import coordinates
from .image import image

__all__ = ["image", "coordinates"]
