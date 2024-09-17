
XRADIO - Xarray Radio Astronomy Data I/O
==========================================

**Docs and code are still under development.**

XRADIO (Xarray Radio Astronomy Data IO) makes working with radio astronomy data in Python simple, efficient, and fun!

The goal of the XRADIO library is to establish a new standard for handling
radio interferometry data, especially visibilities. It is envisioned to act as
a successor to the "measurement set" format (see
e.g. https://casacore.github.io/casacore-notes/229.pdf or
https://casacore.github.io/casacore-notes/264.pdf) that has been used to
represent visibility data in storage.

The main aims of the library are:

* Based on existing Python libraries established in data science, like
  :py:mod:`numpy`, :py:mod:`xarray` and :py:mod:`dask`
* As much as possible compatibility / interoperability with common
  libraries used for atrophysics, like :py:mod:`astropy`.
* Adding well-defined methods for checking schemas
* Covering more use cases as appropriate, such as images or working with
  in-memory data

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Overview

   overview

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   development
   schema

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Correlated Data

   obs_overview
   correlated_data/tutorials/index
   correlated_data/guides/index

   correlated_data/schema_and_api/index


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Image Data

   image_overview

   image_data/tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Design

   decisions


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
