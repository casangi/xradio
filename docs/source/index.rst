XRADIO - Xarray Radio Astronomy Data I/O
========================================

XRADIO (Xarray Radio Astronomy Data I/O) makes working with radio astronomy data in Python simple, efficient, and fun!

XRADIO implements the **Measurement Set v4.0.0** schema, designed for storing radio interferometer and single-dish telescope data for offline processing. Other schemas are being developed.

For a general overview of XRADIO and the schemas included in it, see the section :doc:`Overview <overview>` (it is recommended to pay special attention to the Foundational Reading subsection).
More information on XRADIO development can be found in the section :doc:`Development <development>`.
The Measurement Set v4 is described in the section :doc:`Measurement Set v4.0.0 <measurement_set_overview>`.


Delving into the Measurement Set v4
-----------------------------------

To delve further into the Measurement Set v4,

1. The :doc:`tutorial <measurement_set/tutorials/index>` in the Measurement Set v4.0.0 section demonstrates the schema and API usage.

   - This Jupyter notebook (.ipynb) :doc:`tutorial <measurement_set/tutorials/index>` can be run interactively via the Google Colab link at the top.
   - You can also download and run notebooks locally after installing XRADIO via pip or conda.

2. There are multiple :doc:`guides <measurement_set/guides/index>`:

   - Examples show how different telescopes' data can be represented.
   - If your telescope isn't represented, open an issue and attach a Measurement Set v2 (10MB or smaller).

3. Examine the MSv4 :doc:`schema <measurement_set/schema_and_api/measurement_set_schema>` and :doc:`API documentation <measurement_set/schema_and_api/measurement_set_api.html>` in the Measurement Set v4.0.0 section.

   - The schema is included in ReadTheDocs for versioning and accessibility.

4. The `MS v4 Summary of Testing <https://docs.google.com/document/d/1_r4wKGJx06muO8gV6Yb-xJB7GbCPcGUxTV55l6yytxw/edit?usp=sharing>`_. provides additional information and performance details.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Overview

   overview

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Development

   development
   schema

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Measurement Set v4.0.0

   measurement_set_overview
   measurement_set/tutorials/index
   measurement_set/guides/index

   measurement_set/schema_and_api/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Design

   decisions


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
