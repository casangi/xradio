Introduction
============

.. module:: xradio

XRADIO is an open-source Python package that leverages
`xarray <https://github.com/pydata/xarray>`__ to provide an interface
for radio astronomy data. It provides for each dataset type:

- Documentation and checkers (for documentation and reference for other
  software packages)
- Converters from existing storage formats (such as measurement sets v2)
- Convenience functions for loading, storing and navigating data
- Guides and tutorials

Following `Xarray
terminology <https://docs.xarray.dev/en/latest/user-guide/terminology.html>`__,
data in XRADIO is organized into:

- `Xarray
  DataArrays <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`__:
  a multi-dimensional labeled n-dimensional array
- `Xarray
  Datasets <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`__:
  a container of multiple data arrays, with shared axes and coordinates.
- `Xarray
  DataTrees <https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html>`__:
  a hierarchy of multiple related datasets

This means that XRADIO enables:

- Usage of any storage back-end that is compatible with ``xarray`` (such
  as ``zarr`` and all of its backends)
- Flexible indexing and selection as explained in the `xarray indexing
  and selection
  guide <https://docs.xarray.dev/en/latest/user-guide/indexing.html>`__.
- Native support for lazy loading and distribution using ``dask``.

Dataset types
~~~~~~~~~~~~~

XRADIO is actively developing support for various types of radio
astronomy data:

.. list-table::
   :header-rows: 1
   :widths: 15 5 7 30 15

   * - Dataset type
     - Version
     - Zarr Extension
     - Description
     - Status
   * - :doc:`Measurement Set (v4) <measurement_set/overview>`
     - |MSV4_SCHEMA_VERSION|
     - ``.ps.zarr``
     - Interferometer data (Visibilities) and Single Dish data (Spectrum)
     - Released
   * - Sky and Aperture Images
     - 0.0.0
     - ?
     - Representation of celestial objects and antenna patterns
     - Schema design in progress
   * - Calibration Data
     - 0.0.0
     - ?
     - Information for instrument calibration
     - Schema design in progress
   * - Aperture Models
     - –
     - –
     - Antenna dish models using Zernike polynomials
     - Work scheduled
   * - Simulation Component Lists
     - –
     - –
     - Data for simulating radio astronomy observations
     - Work scheduled

Additional data types will be added based on community needs and
contributions.

Schema Versioning
~~~~~~~~~~~~~~~~~

Each schema in XRADIO follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Changes to existing dimensions, coordinates, data
  variables, attributes, or measures. (This will not occur without wider
  community consultation.)
- **MINOR**: Addition of new datasets. (Backward compatible)
- **PATCH**: Addition of new coordinates, data variables, attributes, or
  measures to existing datasets. (Backward compatible)

For example:

- v4.0.0 to v5.0.0: Major change in the Measurement Set structure
- v4.0.0 to v4.1.0: Addition of a new dataset type
- v4.0.0 to v4.0.1: Addition of new attributes to an existing dataset

The Measurement Set schema starts at v4.0.0, building upon the work of
`Measurement Set
V2 <https://casacore.github.io/casacore-notes/229.pdf>`__ and
`Measurement Set
v3 <https://casacore.github.io/casacore-notes/264.pdf>`__ (which was
never fully implemented).

An XRADIO release will be tied to specific versions of each available
schema. All generated data will include both the XRADIO version and the
schema version in the attribute section.

Installation
~~~~~~~~~~~~

XRADIO can be installed in virtual environments via pip. It is
recommended to use the conda environment manager from
`miniforge <https://github.com/conda-forge/miniforge>`__ to create a
clean, self-contained runtime where XRADIO and all its dependencies can
be installed:

.. code:: sh

   conda create --name xradio python=3.13 --no-default-packages
   conda activate xradio

XRADIO can now be installed using:

.. code:: sh

   pip install xradio

This will also install the minimal dependencies for XRADIO. Note that if
only the minimal dependencies are installed, the functionality to
convert MSv2 to MSv4 will not be available. This requires installing
``python-casacore`` (also included in the ``all`` group, see below), or
alternatively the ``casatools`` backend, as explained in the `casatools
I/O backend guide <measurement_set/guides/backends.md>`__. On macOS it is 
required to install ``python-casacore`` using ``conda install -c conda-forge python-casacore``.

To install the minimal dependencies and the interactive components
(JupyterLab) use:

.. code:: sh

   pip install "xradio[interactive]"

To enable conversion from MSv2 to MSv4 use (this only works for Linux):

.. code:: sh

   pip install "xradio[python-casacore]"

To be able to run tests:

.. code:: sh

   pip install "xradio[test]"

Multiple-dependencies can be installed using:

.. code:: sh

   pip install "xradio[interactive,python-casacore,test]"

To install a more complete set of dependencies:

.. code:: sh

   pip install "xradio[all]"

This will include the dependencies required to run the interactive
Jupyter notebooks, run tests, build documentation, and python-casacore
to enable MSv2=>MSv4 functionality.
