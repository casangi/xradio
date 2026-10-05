
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

Extension Datasets
------------------

Applications can store their own datasets alongside xradio datasets, for
example calibration solutions next to the measurement sets of a processing
set. Such datasets set their ``type`` attribute to an *extension type* of the
form::

    extension:<type>.<namespace>

with the most specific component first:

* ``<type>`` names the kind of dataset, and may have sub-kinds
  (``extension:delay.gains.quartical``).
* ``<namespace>`` is a name controlled by the producer, normally its package
  name. Where that is not unique enough, append a domain in normal DNS order
  (``extension:gains.quartical.sarao.ac.za``).

Components consist of lower-case letters, digits and underscores, start with a
letter, and are separated by ``.``; at least two components are required (see
:py:func:`~xradio.schema.check.is_extension_type`). The type should not
encode a version: put it in a separate ``schema_version`` attribute. Equally,
provenance such as the producing group or organisation belongs in ordinary
attributes, so that it can change without changing the type.

:py:func:`~xradio.schema.check.check_datatree` skips extension datasets
unless a schema has been registered for their type, issuing an
:py:class:`~xradio.schema.check.ExtensionTypeWarning` for each one it skips,
and reports malformed extension types as issues. Children of an extension dataset are still checked
against their own types. The processing set accessor and
:py:func:`~xradio.measurement_set.load_processing_set` ignore extension
datasets stored next to measurement sets.

To have xradio check an extension dataset, a package can declare a schema
for it in the usual way. Importing the module registers the type:

.. code-block:: python

    from typing import Literal

    from xradio.schema.bases import xarray_dataset_schema
    from xradio.schema.typing import Attr, Coord, Data

    @xarray_dataset_schema
    class GainsXds:
        """QuartiCal gain solutions"""

        antenna_name: Coord[Literal["antenna_name"], str]
        gains: Data[Literal["antenna_name"], complex]
        type: Attr[Literal["extension:gains.quartical"]] = "extension:gains.quartical"
        schema_version: Attr[str] = "1.0.0"

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
