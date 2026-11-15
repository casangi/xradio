
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
datasets stored next to measurement sets when computing summaries or applying
selections, but keep them in the processing set: for example,
:py:meth:`~xradio.measurement_set.ProcessingSetXdt.query`
returns them alongside the selected measurement sets.

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

Shared Dimensions
-----------------

.. automodule:: xradio.schema.dimensions

Shared Polarization Vocabulary
------------------------------

.. automodule:: xradio.schema.polarization

.. autodata:: xradio.schema.polarization.CASACORE_STOKES_TYPES
   :no-value:

.. autodata:: xradio.schema.polarization.CANONICAL_POLARIZATION_ORDER

.. autodata:: xradio.schema.polarization.CIRCULAR_CORRELATIONS

.. autodata:: xradio.schema.polarization.LINEAR_CORRELATIONS

.. autofunction:: xradio.schema.polarization.check_polarization_order

The casacore translation tables of :py:mod:`xradio.schema.measures`:

.. autodata:: xradio.schema.measures.CASACORE_SPECTRAL_FRAMES

.. autodata:: xradio.schema.measures.CASACORE_TO_OBSERVER

.. autodata:: xradio.schema.measures.CASACORE_EPOCH_REF_TO_SCALE

.. autodata:: xradio.schema.measures.CASACORE_DIRECTION_TO_FRAME

.. autodata:: xradio.schema.measures.DEFAULT_EQUINOX

Shared Base Classes
-------------------

.. automodule:: xradio.schema.coordinates

.. autoclass:: xradio.schema.coordinates.TimeCoordArrayBase()

.. autoclass:: xradio.schema.coordinates.FrequencyCoordArrayBase()

.. autoclass:: xradio.schema.coordinates.AntennaNameArrayBase()

.. autoclass:: xradio.schema.coordinates.BaselineArrayBase()

.. autoclass:: xradio.schema.coordinates.BaselineAntennaNameArrayBase()

.. autoclass:: xradio.schema.coordinates.ScanArrayBase()

.. autoclass:: xradio.schema.coordinates.ReceptorLabelArrayBase()

.. autoclass:: xradio.schema.measures.TimeSamplingArrayBase()

.. autoclass:: xradio.schema.measures.FrequencySamplingArrayBase()

.. automodule:: xradio.schema.info

.. autoclass:: xradio.schema.info.CreatorDictBase()

.. autoclass:: xradio.schema.info.GenericDataGroupDict()

.. autoclass:: xradio.schema.info.SchemaVersionAttrs()

.. autoclass:: xradio.schema.info.VersionedArrayBase()

.. autoclass:: xradio.schema.info.FlagArrayBase()

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

.. note::

   A :py:class:`~xradio.schema.metamodel.DatasetSchema` gives the
   dimensions that every dataset has in ``dimensions`` (a list holding one
   list) and the dimensions that it may have in addition, in any
   combination, in ``optional_dimensions``. Earlier releases listed every
   combination in ``dimensions`` instead (32 for the image dataset), so the
   ``dimensions`` of ``schemas/ImageXds.json`` changed form. The main
   measurement set datasets have no optional dimensions, so
   ``schemas/VisibilityXds.json`` and ``schemas/SpectrumXds.json`` did not
   change. Schemas exported in the earlier form still import, and check
   the same datasets.

Import and Export
-----------------

.. automodule:: xradio.schema.export
  :members:
