
Data Model Schema
=================

Data model schemas allow us to check whether :py:class:`xarray.DataArray` and
:py:class:`xarray.Dataset` objects conform to the :py:mod:`xradio` schemas (see
e.g. :py:class:`xradio.vis.schema.VisibilityXds`). The approach was essentially
copied from https://pypi.org/project/xarray-dataclasses/, though our
implementation differs in a number of critical ways:

* We use custom decorators on the classes instead of base classes. This
  especially overrides the existing constructor, which makes it easier to
  directly construct instances and allows for extra data variables
  and attributes.

* We support multiple options for types and dimensions

* We convert the schema definition into our own meta-model, which facilitates
  generating documentation generation using Sphinx

Decorators
----------

.. automodule:: xradio.schema

   .. autoclass:: xarray_dataarray_schema

   .. autoclass:: xarray_dataset_schema

   .. autoclass:: dict_schema

.. automodule:: xradio.schema.typing

   .. autoclass:: Data

   .. autoclass:: Dataof

   .. autoclass:: Coord

   .. autoclass:: Coordof
                  
   .. autoclass:: Attr
                  
