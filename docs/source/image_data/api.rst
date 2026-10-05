API documentation
=================

.. automodule:: xradio.image

.. autofunction:: open_image

.. autofunction:: load_image

.. autofunction:: write_image

.. autofunction:: make_empty_sky_image

.. autofunction:: make_empty_aperture_image

.. autofunction:: make_empty_lmuv_image

To check an image dataset against the schema, use
:py:func:`~xradio.image.schema.check_image` (see the :doc:`Image Schema <schema>`).

ImageXds API
------------

Custom accessor to image dataset additional functionality. Given an image :py:class:`xarray.Dataset`, named `img_xds`,
the accessor can be used as `img_xds.xr_img` (`xr` for xradio and `img` for image).

   .. autoclass:: xradio.image.image_xds.ImageXds
      :members:
