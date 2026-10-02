
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

Shared Measures
---------------

.. automodule:: xradio.schema.measures

The quantities and measures in this module are documented with the other
measures (see :ref:`measures`), except
:py:class:`~xradio.schema.measures.PolarizationArray`, which is documented
with the measurement set coordinates. They can also be imported from
``xradio.measurement_set.schema``, where earlier releases defined them.

.. note::

   Because of this move, the ``schema_name`` of five array schemas changed
   from ``xradio.measurement_set.schema.<name>`` to
   ``xradio.schema.measures.<name>`` in the JSON export of the measurement
   set schemas (``schemas/VisibilityXds.json`` and
   ``schemas/SpectrumXds.json``): ``DopplerArray``, ``PolarizationArray``,
   ``QuantityInHertzArray``, ``QuantityInSecondsArray`` and
   ``SpectralCoordArray``. The move does not change what the schemas
   accept, but consumers of the JSON files that look up array schemas by
   ``schema_name`` need the new names. In the same release
   :py:data:`~xradio.schema.measures.AllowedSpectralCoordFrames` gained the
   casacore frames ``GALACTO``, ``LGROUP`` and ``CMB``, so spectral
   coordinates of measurement sets accept these observers too.

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
