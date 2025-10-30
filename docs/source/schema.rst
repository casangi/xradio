
Schema Support
==============

Data model schemas not only allow us to generate documentation,
but also check automatically whether :py:class:`xarray.DataArray` and
:py:class:`xarray.Dataset` objects conform to the :py:mod:`xradio` schemas (see
e.g. :py:class:`xradio.measurement_set.schema.VisibilityXds`). 

Checking
--------

.. automodule:: xradio.schema.check
  :members:

Decorators
----------

.. automodule:: xradio.schema.bases
  :members:

Annotations
-----------

.. automodule:: xradio.schema.typing
  :members:

Data Model
----------

.. automodule:: xradio.schema.metamodel
  :members:

Import and Export
-----------------

.. automodule:: xradio.schema.export
  :members:
