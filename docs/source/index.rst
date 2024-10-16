XRADIO - Xarray Radio Astronomy Data I/O
==========================================

XRADIO (Xarray Radio Astronomy Data I/O) makes working with radio astronomy data in Python simple, efficient, and fun!

Currently, XRADIO implements a draft of the **Measurement Set v4.0.0** schema, designed for storing radio interferometer and single-dish telescope data for offline processing.

Measurement Set v4.0.0 Draft Review
-----------------------------------

The Radio Astronomy Community is invited to review the draft **Measurement Set v4.0.0** schema and reference implementation in XRADIO from October 16 - November 11, 2024.

Providing Feedback
^^^^^^^^^^^^^^^^^^

1. Submit an issue to the `XRADIO GitHub repository <https://github.com/casangi/xradio/issues>`_.
2. Add the label ``MSv4 Review`` to your issue.
3. Your feedback can address the schema, API, bugs, or documentation. We are particularly interested in suggestions for data and meta-data we should include in the schema that will be used in offline processing.

A panel will review all feedback and provide a comprehensive report.

Review Timeline
^^^^^^^^^^^^^^^

- **October 16, 2024**: Documentation release and community notification.
- **November 11, 2024**: Deadline for feedback via GitHub issues.
- **November 25, 2024**: Panel input for final agenda.
- **December 9, 2024**: Review meeting (3 half-days).
- **December 20, 2024**: Submission of review panel report.

The report will be used by the XRADIO team to finalize the Measurement Set v4.0.0 schema and API.

Conducting a Comprehensive Review
---------------------------------

To thoroughly review XRADIO and the Measurement Set v4.0.0 draft:

1. Read the XRADIO :doc:`Overview <overview>`, :doc:`Development <development>`, and :doc:`Measurement Set v4.0.0 <measurement_set_overview>` sections.

   - Pay special attention to the Foundational Reading subsection in the Overview.

2. Complete the :doc:`tutorial <measurement_set/tutorials/index>` in the Measurement Set v4.0.0 section, which demonstrates the schema and API usage.

   - This Jupyter notebook (.ipynb) :doc:`tutorial <measurement_set/tutorials/index>` can be run interactively via the Google Colab link at the top.
   - You can also download and run notebooks locally after installing XRADIO via pip.

3. Review :doc:`guides <measurement_set/guides/index>` relevant to your interests.

   - Examples show how different telescopes' data can be represented.
   - If your telescope isn't represented, open an issue and attach a Measurement Set v2 (10MB or smaller).

4. Examine the [schema](https://xradio.readthedocs.io/en/latest/measurement_set/schema_and_api/measurement_set_schema.html) and [API documentation](https://xradio.readthedocs.io/en/latest/measurement_set/schema_and_api/measurement_set_api.html) in the Measurement Set v4.0.0 section.

   - The schema is included in ReadTheDocs for versioning and accessibility.

5. Read the [MSv4 Summary of Testing](https://docs.google.com/document/d/1_r4wKGJx06muO8gV6Yb-xJB7GbCPcGUxTV55l6yytxw/edit?usp=sharing).

We appreciate your participation in this review process and look forward to your valuable input.

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
