Conventions
===========

These conventions apply to all XRADIO schemas.

Time and frequency order
------------------------

The ``time`` and ``frequency`` coordinates must always increase. Their step does not have to be constant, but a value
cannot repeat.

XRADIO handles data stored in another order as follows:

- :py:func:`~xradio.measurement_set.convert_msv2_to_processing_set` always gives increasing times, and sorts the
  frequencies of a spectral window. A repeated frequency is an error.
- The lazy readers reverse a decreasing axis on the fly: :py:func:`~xradio.measurement_set.open_msv2`, ``open_asdm``,
  :py:func:`~xradio.image.open_image` for CASA and FITS images, and :py:func:`~xradio.image.load_image` for CASA images.
- These readers raise an error for an axis that neither increases nor decreases. Convert the data, or fix the order with
  CASA: for a Measurement Set, ``split`` or ``mstransform`` with the channel ranges in frequency order. Images cannot be
  converted: fix their spectral axis, for example with CASA.

Datasets created in other ways, and Processing Sets and zarr images already stored, are not checked.
