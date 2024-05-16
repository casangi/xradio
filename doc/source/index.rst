
Xradio - Xarray-style radio astronomy data
==========================================

The goal of the Xradio library is to establish a new standard for handling
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
   :maxdepth: 1
   :caption: Getting Started

   vis_tutorial
   image_tutorial

.. toctree::
   :maxdepth: 1
   :caption: Guides

   lofar_conversion
   meerkat_conversion
   meta_data_proposal
   ska_mid_conversion

.. toctree::
   :maxdepth: 1
   :caption: Design

   decisions

.. toctree::
   :maxdepth: 1
   :caption: Reference

   vis_api
   vis_model


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
