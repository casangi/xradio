Schemas
=======

Definitions not specific to a particular dataset schema.

.. \_quantities:

Quantites
---------

Quantity
`DataArrays <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`__
encode general-purpose data that relates to certain physical quantities.
They are typically associated with an SI unit.

.. autoclass:: xradio.measurement_set.schema.QuantityInSecondsArray()
   :members:
   :show-inheritance:

.. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInSecondsArray

.. autoclass:: xradio.measurement_set.schema.QuantityInHertzArray()
   :members:
   :show-inheritance:

.. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInHertzArray

.. autoclass:: xradio.measurement_set.schema.QuantityInMetersArray()
   :members:
   :show-inheritance:

.. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInMetersArray

.. autoclass:: xradio.measurement_set.schema.QuantityInMetersPerSecondArray()
   :members:
   :show-inheritance:

.. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInMetersPerSecondArray

.. autoclass:: xradio.measurement_set.schema.QuantityInRadiansArray()
   :members:
   :show-inheritance:

.. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInRadiansArray

.. autoclass:: xradio.measurement_set.schema.QuantityInKelvinArray()
   :members:
   :show-inheritance:

.. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInKelvinArray

.. autoclass:: xradio.measurement_set.schema.QuantityInKelvinPerJanskyArray()
   :members:
   :show-inheritance:

.. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInKelvinPerJanskyArray

.. autoclass:: xradio.measurement_set.schema.QuantityInPascalArray()
   :members:
   :show-inheritance:

.. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInPascalArray

.. autoclass:: xradio.measurement_set.schema.QuantityInPerSquareMetersArray()
   :members:
   :show-inheritance:

.. xradio_array_schema_table:: xradio.measurement_set.schema.QuantityInPerSquareMetersArray

.. \_measures:

Measures
--------

As with `python-casacore
measures <https://casacore.github.io/python-casacore/casacore_measures.html>`__,
measures are quantities that are interpreted in relation to a specified
reference frame (such as UTC for
`TimeArray <measures.rst#xradio.measurement_set.schema.TimeArray>`__
or FK5 for
`SkyCoordArray <measures.rst#xradio.measurement_set.schema.SkyCoordArray>`__).
Measure definitions are aligned with `astropy
coordinate <https://docs.astropy.org/en/stable/coordinates/index.html>`__
naming conventions as much as possible. The table below outlines the
different types of XRADIO measures:

.. autoclass:: xradio.measurement_set.schema.TimeArray()

.. xradio_array_schema_table:: xradio.measurement_set.schema.TimeArray

.. autoclass:: xradio.measurement_set.schema.SpectralCoordArray()

.. xradio_array_schema_table:: xradio.measurement_set.schema.SpectralCoordArray

.. autoclass:: xradio.measurement_set.schema.SkyCoordArray()

.. xradio_array_schema_table:: xradio.measurement_set.schema.SkyCoordArray

.. autoclass:: xradio.measurement_set.schema.LocationArray()

.. xradio_array_schema_table:: xradio.measurement_set.schema.LocationArray

.. autoclass:: xradio.measurement_set.schema.DopplerArray()

.. xradio_array_schema_table:: xradio.measurement_set.schema.DopplerArray

.. raw:: html

   <!-- <iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQRZyrmK41kXbeaq1V7UFK8IDO5u-zIt5I-4xUbxjOX7oK5muw0vFufreSLMn23KOqtawWjkgtGyfTR/pubhtml?gid=1504318014&single=true" 
           width="80%" 
           height="600" 
           frameborder="0" 
           scrolling="no">
   </iframe> -->

Coordinate Labels
-----------------

For some types of measures, the data consists of values that are labeled
using coordinate labels. These labels provide context for interpreting
the data:

========================= ============== =====================
Coordinate Label Name     Values         Related Measures Type
========================= ============== =====================
ellipsoid_dir_label       lon, lat       location
ellipsoid_dis_label       height         location
cartesian_pos_label       x, y, z        location
cartesian_local_pos_label p, q, r        location
galactic_sky_dir_label    lon, lat       sky_coord
local_sky_dir_label       az, alt        sky_coord
local_sky_dis_label       dist           sky_coord
sky_dir_label             ra, dec        sky_coord
sky_dist_label            dist           sky_coord
uvw_label                 u, v, w        uvw
receptor_label            pol_0, pol_1   quantity
tone_label                tone_0, tone_1 spectral_coord
========================= ============== =====================

Measures Example
----------------

The following example illustrates how measures information is included
in both a data variable (``FIELD_PHASE_CENTER``) and a coordinate
(``time``). The ``FIELD_PHASE_CENTER`` data variable has the dimensions
``time`` and ``sky_dir_label``. Note that the ``sky_coord`` measure
requires only the ``sky_dir_label`` dimension, not the ``time``
dimension.

.. code:: ipython3

    import xarray as xr
    phase_center = xr.DataArray()
    
    import numpy as np
    import xarray as xr
    import pandas as pd
    
    # Create an empty Xarray Dataset.
    xds = xr.Dataset()
    
    # Create the time coordinate with time measures attributes.
    time = xr.DataArray(pd.date_range('2000-01-01', periods=3).astype('datetime64[s]').astype(int), dims='time', attrs={'type': 'time', 'units': 's', 'format':'unix', 'scale':'utc'})
    
    # Create FIELD_PHASE_CENTER data variable with coordinates time x sky_dir_label.
    coords = {'time': time,
              'sky_dir_label': ['ra', 'dec']}
    
    data = np.array([[-2.10546176, -0.29611873],
           [-2.10521098, -0.29617315],
           [-2.1050196, -0.2961987]])
    
    xds['FIELD_PHASE_CENTER'] = xr.DataArray(data, coords=coords, dims=['time', 'sky_dir_label'])
    
    # Add sky_coord measures attributes to FIELD_PHASE_CENTER.
    xds['FIELD_PHASE_CENTER'].attrs = {
        "type": "sky_coord",
        "units": "rad",
        "frame": "icrs",
    }
    
    xds




.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
     *
     */
    
    :root {
      --xr-font-color0: var(
        --jp-content-font-color0,
        var(--pst-color-text-base rgba(0, 0, 0, 1))
      );
      --xr-font-color2: var(
        --jp-content-font-color2,
        var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
      );
      --xr-font-color3: var(
        --jp-content-font-color3,
        var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
      );
      --xr-border-color: var(
        --jp-border-color2,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
      );
      --xr-disabled-color: var(
        --jp-layout-color3,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
      );
      --xr-background-color: var(
        --jp-layout-color0,
        var(--pst-color-on-background, white)
      );
      --xr-background-color-row-even: var(
        --jp-layout-color1,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
      );
      --xr-background-color-row-odd: var(
        --jp-layout-color2,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
      );
    }
    
    html[theme="dark"],
    html[data-theme="dark"],
    body[data-theme="dark"],
    body.vscode-dark {
      --xr-font-color0: var(
        --jp-content-font-color0,
        var(--pst-color-text-base, rgba(255, 255, 255, 1))
      );
      --xr-font-color2: var(
        --jp-content-font-color2,
        var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
      );
      --xr-font-color3: var(
        --jp-content-font-color3,
        var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
      );
      --xr-border-color: var(
        --jp-border-color2,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
      );
      --xr-disabled-color: var(
        --jp-layout-color3,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
      );
      --xr-background-color: var(
        --jp-layout-color0,
        var(--pst-color-on-background, #111111)
      );
      --xr-background-color-row-even: var(
        --jp-layout-color1,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
      );
      --xr-background-color-row-odd: var(
        --jp-layout-color2,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
      );
    }
    
    .xr-wrap {
      display: block !important;
      min-width: 300px;
      max-width: 700px;
    }
    
    .xr-text-repr-fallback {
      /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
      display: none;
    }
    
    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
      margin-bottom: 4px;
      border-bottom: solid 1px var(--xr-border-color);
    }
    
    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }
    
    .xr-obj-type,
    .xr-array-name {
      margin-left: 2px;
      margin-right: 10px;
    }
    
    .xr-obj-type {
      color: var(--xr-font-color2);
    }
    
    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
    }
    
    .xr-section-item {
      display: contents;
    }
    
    .xr-section-item input {
      display: inline-block;
      opacity: 0;
      height: 0;
    }
    
    .xr-section-item input + label {
      color: var(--xr-disabled-color);
      border: 2px solid transparent !important;
    }
    
    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }
    
    .xr-section-item input:focus + label {
      border: 2px solid var(--xr-font-color0) !important;
    }
    
    .xr-section-item input:enabled + label:hover {
      color: var(--xr-font-color0);
    }
    
    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
    }
    
    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.5em;
    }
    
    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }
    
    .xr-section-summary-in + label:before {
      display: inline-block;
      content: "►";
      font-size: 11px;
      width: 15px;
      text-align: center;
    }
    
    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-summary-in:checked + label:before {
      content: "▼";
    }
    
    .xr-section-summary-in:checked + label > span {
      display: none;
    }
    
    .xr-section-summary,
    .xr-section-inline-details {
      padding-top: 4px;
      padding-bottom: 4px;
    }
    
    .xr-section-inline-details {
      grid-column: 2 / -1;
    }
    
    .xr-section-details {
      display: none;
      grid-column: 1 / -1;
      margin-bottom: 5px;
    }
    
    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }
    
    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }
    
    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }
    
    .xr-preview {
      color: var(--xr-font-color3);
    }
    
    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }
    
    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }
    
    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }
    
    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }
    
    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }
    
    .xr-dim-list:before {
      content: "(";
    }
    
    .xr-dim-list:after {
      content: ")";
    }
    
    .xr-dim-list li:not(:last-child):after {
      content: ",";
      padding-right: 5px;
    }
    
    .xr-has-index {
      font-weight: bold;
    }
    
    .xr-var-list,
    .xr-var-item {
      display: contents;
    }
    
    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      border-color: var(--xr-background-color-row-odd);
      margin-bottom: 0;
      padding-top: 2px;
    }
    
    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }
    
    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
      border-color: var(--xr-background-color-row-even);
    }
    
    .xr-var-name {
      grid-column: 1;
    }
    
    .xr-var-dims {
      grid-column: 2;
    }
    
    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }
    
    .xr-var-preview {
      grid-column: 4;
    }
    
    .xr-index-preview {
      grid-column: 2 / 5;
      color: var(--xr-font-color2);
    }
    
    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }
    
    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }
    
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      display: none;
      border-top: 2px dotted var(--xr-background-color);
      padding-bottom: 20px !important;
      padding-top: 10px !important;
    }
    
    .xr-var-attrs-in + label,
    .xr-var-data-in + label,
    .xr-index-data-in + label {
      padding: 0 1px;
    }
    
    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data,
    .xr-index-data-in:checked ~ .xr-index-data {
      display: block;
    }
    
    .xr-var-data > table {
      float: right;
    }
    
    .xr-var-data > pre,
    .xr-index-data > pre,
    .xr-var-data > table > tbody > tr {
      background-color: transparent !important;
    }
    
    .xr-var-name span,
    .xr-var-data,
    .xr-index-name div,
    .xr-index-data,
    .xr-attrs {
      padding-left: 25px !important;
    }
    
    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      grid-column: 1 / -1;
    }
    
    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }
    
    .xr-attrs dt,
    .xr-attrs dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }
    
    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }
    
    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }
    
    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }
    
    .xr-icon-database,
    .xr-icon-file-text2,
    .xr-no-icon {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    
    .xr-var-attrs-in:checked + label > .xr-icon-file-text2,
    .xr-var-data-in:checked + label > .xr-icon-database,
    .xr-index-data-in:checked + label > .xr-icon-database {
      color: var(--xr-font-color0);
      filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
      stroke-width: 0.8px;
    }
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 96B
    Dimensions:             (time: 3, sky_dir_label: 2)
    Coordinates:
      * time                (time) int64 24B 946684800 946771200 946857600
      * sky_dir_label       (sky_dir_label) &lt;U3 24B &#x27;ra&#x27; &#x27;dec&#x27;
    Data variables:
        FIELD_PHASE_CENTER  (time, sky_dir_label) float64 48B -2.105 ... -0.2962</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-fcdac2ba-b9c2-4fce-ba02-7d13695897a5' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-fcdac2ba-b9c2-4fce-ba02-7d13695897a5' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 3</li><li><span class='xr-has-index'>sky_dir_label</span>: 2</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-868ef41d-18e5-4865-baa6-2ca9c952da66' class='xr-section-summary-in' type='checkbox'  checked><label for='section-868ef41d-18e5-4865-baa6-2ca9c952da66' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>946684800 946771200 946857600</div><input id='attrs-163c8ac3-d7d5-467c-b506-2b12aa218f73' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-163c8ac3-d7d5-467c-b506-2b12aa218f73' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a74aa71d-2141-4601-b5d4-3c1f9274cf37' class='xr-var-data-in' type='checkbox'><label for='data-a74aa71d-2141-4601-b5d4-3c1f9274cf37' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>type :</span></dt><dd>time</dd><dt><span>units :</span></dt><dd>s</dd><dt><span>format :</span></dt><dd>unix</dd><dt><span>scale :</span></dt><dd>utc</dd></dl></div><div class='xr-var-data'><pre>array([946684800, 946771200, 946857600])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sky_dir_label</span></div><div class='xr-var-dims'>(sky_dir_label)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;ra&#x27; &#x27;dec&#x27;</div><input id='attrs-a0a320ad-b6ec-45d3-a52a-8ad190731e29' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a0a320ad-b6ec-45d3-a52a-8ad190731e29' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-96e03b86-97b7-4c55-8059-c998c8613381' class='xr-var-data-in' type='checkbox'><label for='data-96e03b86-97b7-4c55-8059-c998c8613381' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;ra&#x27;, &#x27;dec&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-aa8d7ad6-1f57-4e8e-a570-c34a747cc5a8' class='xr-section-summary-in' type='checkbox'  checked><label for='section-aa8d7ad6-1f57-4e8e-a570-c34a747cc5a8' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>FIELD_PHASE_CENTER</span></div><div class='xr-var-dims'>(time, sky_dir_label)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-2.105 -0.2961 ... -2.105 -0.2962</div><input id='attrs-9d60d0b7-d990-4880-bf82-600607a73528' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-9d60d0b7-d990-4880-bf82-600607a73528' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-92274900-b8df-488d-88f0-e62cd3999e3d' class='xr-var-data-in' type='checkbox'><label for='data-92274900-b8df-488d-88f0-e62cd3999e3d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>type :</span></dt><dd>sky_coord</dd><dt><span>units :</span></dt><dd>&#x27;rad&#x27;</dd><dt><span>frame :</span></dt><dd>icrs</dd></dl></div><div class='xr-var-data'><pre>array([[-2.10546176, -0.29611873],
           [-2.10521098, -0.29617315],
           [-2.1050196 , -0.2961987 ]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ac55fa92-b90d-49b3-9a46-a589a74065ce' class='xr-section-summary-in' type='checkbox'  ><label for='section-ac55fa92-b90d-49b3-9a46-a589a74065ce' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>time</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-9013b0be-4f31-4096-b66d-ae1c64ca408e' class='xr-index-data-in' type='checkbox'/><label for='index-9013b0be-4f31-4096-b66d-ae1c64ca408e' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([946684800, 946771200, 946857600], dtype=&#x27;int64&#x27;, name=&#x27;time&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>sky_dir_label</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-c44a10b5-2ad6-46ee-a655-bbbc37b51712' class='xr-index-data-in' type='checkbox'/><label for='index-c44a10b5-2ad6-46ee-a655-bbbc37b51712' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;ra&#x27;, &#x27;dec&#x27;], dtype=&#x27;object&#x27;, name=&#x27;sky_dir_label&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-798700cb-0da4-4def-9e38-90fed77365dc' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-798700cb-0da4-4def-9e38-90fed77365dc' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



.. code:: ipython3

    # Example of creating an Astropy SkyCoord object from the FIELD_PHASE_CENTER data variable.
    from astropy.coordinates import SkyCoord
    astropy_skycoord = SkyCoord(ra=xds.FIELD_PHASE_CENTER.sel(sky_dir_label='ra').values,dec=xds.FIELD_PHASE_CENTER.sel(sky_dir_label='dec').values,unit='rad',frame=xds.FIELD_PHASE_CENTER.attrs['frame'])
    astropy_skycoord




.. parsed-literal::

    <SkyCoord (ICRS): (ra, dec) in deg
        [(239.36592723, -16.96635346), (239.38029586, -16.9694715 ),
         (239.39126113, -16.97093541)]>


