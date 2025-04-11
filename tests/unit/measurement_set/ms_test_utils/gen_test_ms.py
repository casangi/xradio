"""
Generates MeasurementSets. The motivation is to generate on-the-fly MSs in
unit tests (normally as pytest fixtures).
It has been growing driven by iterative increase of coverage in unit tests.
"""

from contextlib import contextmanager
import datetime
import itertools
from pathlib import Path
from typing import Generator

import numpy as np

import casacore.tables as tables
from casacore.tables import default_ms, default_ms_subtable
from casacore.tables.tableutil import makedminfo, maketabdesc
from casacore.tables.msutil import complete_ms_desc, makearrcoldesc, required_ms_desc


# 2 observations, 2 fields, 2 states
# 2 SPWs, 4 polarizations
default_ms_descr = {
    "nchans": 32,
    "npols": 2,
    "data_cols": ["DATA"],  # ['CORRECTED_DATA'],
    # DATA / CORRECTED, etc.
    # subtables needed to test xds structure
    "SPECTRAL_WINDOW": {"0": 0, "1": 1},
    "POLARIZATION": {"0": 0, "1": 1},  # "2": 2, "3": 3},
    "ANTENNA": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
    # subtables neded to test partitioning
    "FIELD": {"0": 0, "1": 1},
    "EPHEMERIDES": {0: 0},
    "SCAN": {
        "1": {
            "0": {"intent": "CAL_ATMOSPHERE#ON_SOURCE"},
            "1": {"intent": "CAL_ATMOSPHERE#OFF_SOURCE"},
        },
        "2": {
            "0": {"intent": "OBSERVE_TARGET#ON_SOURCE"},
            "1": {"intent": "OBSERVE_TARGET#ON_SOURCE"},
            "2": {"intent": "OBSERVE_TARGET#ON_SOURCE"},
            "3": {"intent": "OBSERVE_TARGET#ON_SOURCE"},
        },
        "3": {
            "0": {"intent": "CALIBRATE_DELAY#ON_SOURCE,CALIBRATE_PHASE#ON_SOURCE"},
            "1": {"intent": "CALIBRATE_DELAY#ON_SOURCE,CALIBRATE_PHASE#ON_SOURCE"},
            "2": {"intent": "CALIBRATE_DELAY#ON_SOURCE,CALIBRATE_PHASE#ON_SOURCE"},
        },
    },
    # CALIBRATE_ATMOSPHERE#OFF_SOURCE,CALIBRATE_ATMOSPHERE#ON_SOURCE,CALIBRATE_WVR#OFF_SOURCE,CALIBRATE_WVR#ON_SOURCE
    "STATE": {
        "0": {"id": 0, "intent": "CAL_ATMOSPHERE#ON_SOURCE"},
        "1": {"id": 1, "intent": "CAL_ATMOSPHERE#OFF_SOURCE"},
    },
    "OBSERVATION": {"0": 0, "1": 1},
    # Mandatory as per MSv2 but not essential to be able to build xdss
    "FEED": {"0": 0},
    "POINTING": {},
    "PROCESSOR": {"0": 0, "1": 1},
    "SOURCE": {"0": 0, "1": 1},
    # Auto-generated without parameters for now:
    # 'FLAG_CMD': {},
    # 'HISTORY': {},
}


def gen_test_ms(
    msname: str,
    descr: dict = None,
    opt_tables: bool = True,
    required_only: bool = True,
    misbehave: bool = False,
):
    """
    Generates an MS for testing purposes, including main table and
    subtables, either only the required ones or all. Follows the MSv2
    definitions. The aim is to produce MSs with the minimal structure and
    set of rows to exercise xradio functions.

    - To be able to effectively start testing some functions, these are
    required: MAIN, DATA_DESCRIPTION, SPECTRAL_WINDOW, POLARIZATION, ANTENNA
    - To be able to start testing partitioning: FIELD, STATE
    - Furhter partitioning testing: observation, processor, feed?

    - The optional tables SYSCAL, WEATHER are generated with very (too)
      simple contents

    Parameters
    ----------
    msname : str
        name of MS on disk
    descr : dict (Default value = None)
        MS description, including description of scans, fields,
        intents, etc. By default (empty dict) will create an MS with minimal
        structure (one observation, one array, one scan, one field, etc.)
    opt_tables : bool (Default value = True)
        whether to produce optional (sub)tables, such as SOURCE, WEATHER
    required_only : bool (Default value = True)
        whether to use the complete or required columns spec
    misbehave : bool (Default value = False)
        whether to generate a misbehaving MS. For example, usual or more
        corner case conformance issues such as absence of STATE subtable,
        missing FEED subtable, presence of missing SOURCE_IDs in the FIELD
        subtable, an ASDM_EXECBLOCK table that is empty, etc.

    Returns
    -------

    """
    if not descr:
        descr = default_ms_descr

    # Definitely needed: main table + following subtables: DATA_DESCRIPTPION
    #  + SPECTRAL_WINDOW + POLARIZATION
    #  + ANTENNNA (just for the names)
    # All these are required to create the xdss (they define data and dims)
    outdescr = gen_main_table(msname, descr, required_only)
    gen_subt_ddi(msname, descr["SPECTRAL_WINDOW"], descr["POLARIZATION"])
    gen_subt_spw(msname, descr["SPECTRAL_WINDOW"])
    gen_subt_antenna(msname, descr["ANTENNA"])
    gen_subt_pol_setup(msname, descr["npols"], descr["POLARIZATION"])

    gen_ephem = not misbehave
    # Also needed for partitioning: FIELD, STATE
    gen_subt_field(msname, descr["FIELD"], gen_ephem=gen_ephem, misbehave=misbehave)
    gen_subt_state(msname, descr["STATE"])

    # Required by MSv2/v3 but not strictly required to load and partition:

    # BEAM (only MSv3)

    # FEED
    gen_subt_feed(
        msname, descr["FEED"], descr["ANTENNA"], descr["SPECTRAL_WINDOW"], misbehave
    )

    # HISTORY
    gen_subt_history(msname, descr["OBSERVATION"])

    # OBSERVATION
    gen_subt_observation(msname, descr["OBSERVATION"])

    # POINTING: TODO
    gen_subt_pointing(msname)

    # PROCESSOR
    gen_subt_processor(msname, descr["PROCESSOR"])

    if opt_tables:
        # DOPPLER: no examples seen in test MSs

        # EPHEMERIDES (only defined in Msv3, never seen in practice)

        # FLAG_CMD (made optional in MSv3, required in MSv2)
        gen_subt_flag_cmd(msname)

        # FREQ_OFFSET: no examples seen in test MSs

        # INTERFEROMETER_MODEL (only MSv3)

        # PHASED_ARRAY (only MSv3)

        # QUALITY_FREQUENCY_STATISTIC (only MSv3: listend but never defined)

        # QUALITY_BASELINE_STATISTIC (only MSv3: listend but never defined)

        # QUALITY_TIME_STATISTIC (only MSv3: listend but never defined)

        # SOURCE
        gen_subt_source(msname, descr["SOURCE"], misbehave)

        # SCAN (Only MSv3)

        # SYSCAL
        gen_subt_syscal(msname, descr["ANTENNA"])

        # WEATHER
        gen_subt_weather(msname)

        # ASDM_* subtables. One simple example
        gen_subt_asdm_receiver(msname)
        # ASDM_EXECBLOCK is used in create_info_dicts when available
        gen_subt_asdm_execblock(msname, misbehave)

    return outdescr


def make_ms_empty(name: str, descr: dict = None, complete: bool = False):
    """
    OLD simple function. makes empty (0 rows) MSs

    Parameters
    ----------
    name : str

    descr : dict (Default value = None)

    complete : bool (Default value = False)

    Returns
    -------

    """
    if complete:
        tabdesc = complete_ms_desc("MAIN")
    else:
        tabdesc = required_ms_desc("MAIN")

    datacoldesc = tables.makearrcoldesc(
        "DATA",
        0.1 + 0.1j,
        valuetype="complex",
        ndim=2,
        datamanagertype="TiledShapeStMan",
        datamanagergroup="TiledData",
        comment="The data column",
    )
    del datacoldesc["desc"]["shape"]
    tabdesc.update(tables.maketabdesc(datacoldesc))

    weightspeccoldesc = tables.makearrcoldesc(
        "WEIGHT_SPECTRUM",
        1.0,
        valuetype="float",
        ndim=2,
        datamanagertype="TiledShapeStMan",
        datamanagergroup="TiledWgtSpectrum",
        comment="Weight for each data point",
    )
    del weightspeccoldesc["desc"]["shape"]
    tabdesc.update(tables.maketabdesc(weightspeccoldesc))

    vis = tables.default_ms(name, tabdesc=tabdesc, dminfo=makedminfo(tabdesc))
    assert vis.nrows() == 0


def gen_main_table(mspath: str, descr: dict, required_only: bool = True):
    """
    Create main MSv2 table.
    Relies on the required/complete_ms_desc descriptions of columns

    Simplifications/assumptions:
    - All polarization setups have the same number of correlations/
    - All scans have the same SPWs (all SPWs)
    - Multiple observation: just repeat the same structure (same scans,
      fields, etc.)
    - Can generate multiple *_DATA columns but they all have the same data
      pattern

    :return: outdescr

    Parameters
    ----------
    mspath: str :

    descr: dict :

    required_only: bool :
         (Default value = True)

    Returns
    -------

    """

    outdescr = descr.copy()

    if descr == {}:
        nchans = 16
        npols = 2
    else:
        nchans = descr["nchans"]
        npols = descr["npols"]

    if required_only:
        ms_desc = required_ms_desc("MAIN")
    else:
        ms_desc = complete_ms_desc("MAIN")

    ms_desc["UVW"].update(
        options=0,
        shape=[3],
        ndim=1,
        dataManagerGroup="UVWGroup",
        dataManagerType="TiledColumnStMan",
    )
    dmgroups_spec = {"UVW": {"DEFAULTTILESHAPE": [3, nchans]}}

    # data columns. TODO: move to function - make_data_cols_descr
    for data_col_name in descr["data_cols"]:
        data_col_desc = makearrcoldesc(
            data_col_name,
            0.0,
            options=4,
            valuetype="complex",
            # Beware of the additional column description entry - not present in casatestdata
            shape=[nchans, npols],
            ndim=2,
            datamanagertype="TiledColumnStMan",
            datamanagergroup="DataGroup",
            comment="added by gen_test_ms",
        )
        ms_desc.update(maketabdesc(data_col_desc))

        dmgroups_spec.update(
            {"DataColsGroup": {"DEFAULTTILESHAPE": [nchans, npols, 32]}}
        )
        ms_data_man_info = makedminfo(ms_desc, dmgroups_spec)

    if not "STATE" in descr:
        vis = tables.default_ms(name, tabdesc=tabdesc, dminfo=makedminfo(tabdesc))
        assert vis.nrows() == 0
        return
    # else:
    #    populate (TODO)

    # Key columns (that define data rows):
    # - TIME: given in descr
    # - ANTENNA1, ANTENNA2: given in descr
    # - DATA_DESC_ID, PROCESSOR_ID: given in descr
    # - FEED1, FEED2, assumed 0
    # - Not considered: ANTENNA3, FEED3, PHASE_ID, TIME_EXTRA_PREC

    with default_ms(mspath, ms_desc, ms_data_man_info) as msv2:
        desc = msv2.getcoldesc("UVW")
        assert desc["dataManagerType"] == "TiledColumnStMan"
        dminfo = msv2.getdminfo("UVW")
        assert dminfo["NAME"] == "UVWGroup"

        # Figure out amount of rows and related IDs
        nrows = 300
        dd_pairs = list(
            itertools.product(
                list(descr["SPECTRAL_WINDOW"].values()), descr["POLARIZATION"].values()
            )
        )
        nddis = len(dd_pairs)
        outdescr["nddis"] = nddis
        dd_ids = np.arange(nddis)
        nrows *= nddis
        # problem: TIME - SPW

        msv2.addrows(nrows)
        # trasnposing so that indices increase with channel number, not pol
        vis_val = np.arange(nchans * npols).reshape(
            (npols, nchans)
        ).transpose() * np.complex64(1 - 1j)
        # vis_val *= np.arange(
        # msv2.putcell("CORRECTED_DATA", 0, vis_val)
        for data_col in descr["data_cols"]:
            msv2.putcol(data_col, np.broadcast_to(vis_val, (nrows, nchans, npols)))

        # === Key attributes ===
        # Make Ids

        # TIME
        CASACORE_TO_DATETIME_CORRECTION = 3_506_716_800.0
        start = (
            datetime.datetime(
                2023, 5, 1, 1, 1, tzinfo=datetime.timezone.utc
            ).timestamp()
            + CASACORE_TO_DATETIME_CORRECTION
        )
        time_col = np.arange(nrows) + start
        msv2.putcol("TIME", time_col)

        # (TIME_EXTRA_PREC): nothing for now

        nants = len(descr["ANTENNA"])
        # ANTENNA1, ANTENNA2
        # baselines = list(itertools.product(ants, ants))
        # combinations without repetitions: baselines as list of tuples
        ants = np.arange(nants)
        baselines = list(itertools.combinations(ants, 2))
        ant1_ant2 = list(zip(*baselines))

        # TODO: needs fixes for rounding
        reps = np.tile(ant1_ant2[0], int(nrows / len(baselines)))
        msv2.putcol("ANTENNA1", np.tile(ant1_ant2[0], int(nrows / len(baselines))))
        msv2.putcol("ANTENNA2", np.tile(ant1_ant2[1], int(nrows / len(baselines))))

        # (ANTENNA3): nothing for now

        msv2.putcol("FEED1", np.broadcast_to(0, (nrows)))

        msv2.putcol("FEED2", np.broadcast_to(0, (nrows)))

        # (FEED3): nothing for now

        # DATA_DESC_ID
        # msv2.putcol("DATA_DESC_ID", np.broadcast_to(0, (nrows)))
        ddi_col = np.repeat(dd_ids, nrows / nddis)
        # print(f" This is nddis: {nddis} len dd_id: {len(dd_ids)}, with nrows: {nrows}")
        msv2.putcol("DATA_DESC_ID", np.broadcast_to(ddi_col, (nrows)))

        msv2.putcol("PROCESSOR_ID", np.broadcast_to(0, (nrows)))

        msv2.putcol("FIELD_ID", np.broadcast_to(0, (nrows)))

        # PHASE_ID: nothing for now

        # === Non-key attributes ===

        # The ones that are IDs
        msv2.putcol("OBSERVATION_ID", np.broadcast_to(0, (nrows)))

        msv2.putcol("ARRAY_ID", np.broadcast_to(0, (nrows)))

        msv2.putcol("SCAN_NUMBER", np.broadcast_to(1, (nrows)))

        # STATE_ID: if no states/intents => all STATE_ID = -1
        no_state = False
        if no_state:
            msv2.putcol("STATE_ID", np.broadcast_to(-1, (nrows)))
        else:
            msv2.putcol("STATE_ID", np.broadcast_to(0, (nrows)))

        # Make other scalar / inc columns

        interval = 1.0
        msv2.putcol("INTERVAL", np.broadcast_to(interval, (nrows)))

        exposure = 0.9 * interval
        msv2.putcol("EXPOSURE", np.broadcast_to(exposure, (nrows)))

        # TIME_CENTROID
        msv2.putcol("TIME_CENTROID", time_col)

        # (PULSAR_BIN)

        # (PULSAR_GATE_ID)

        # (BASELINE_REF)

        # UVW
        msv2.putcol("UVW", np.broadcast_to([1.0, 0.5, 1.5], (nrows, 3)))

        # (UVW2)

        # (DATA)

        # (FLOAT_DATA)

        # (VIDEO_POINT)

        # (LAG_DATA)

        # SIGMA
        msv2.putcol("SIGMA", np.broadcast_to(1.0, (nrows, npols)))

        # (SIGMA_SPECTRUM)

        # WEIGHT
        # msv2.putcol("WEIGHT", np.broadcast_to(1.0, (nrows, npols)))

        # (WEIGHT_SPECTRUM)

        msv2.putcol("FLAG", np.broadcast_to(False, (nrows, nchans, npols)))

        # FLAG_CATEGORY: leave empty

        msv2.putcol("FLAG_ROW", np.broadcast_to(False, (nrows)))

    # with tables.table(mspath) as main:
    #     print(f"** after default_ms: {main.getkeywords()}")
    return outdescr


def gen_subt_ddi(mspath: str, spws_descr: dict, pol_setup_descr: dict):
    """
    Populates DATA_DESCRIPTION
    Q: spws_descr with IDs in keys? IDs in MSv2 are implicit 0...n-1
    """
    import itertools

    # TODO generate spw-pol
    with tables.table(
        mspath + "::DATA_DESCRIPTION", ack=False, readonly=False
    ) as ddi_tbl:
        nrows = len(spws_descr) * len(pol_setup_descr)
        ddi_tbl.addrows(nrows)

        ddis = list(
            itertools.product(list(spws_descr.values()), pol_setup_descr.values())
        )
        spw_ids = list(zip(*ddis))[0]
        pol_ids = list(zip(*ddis))[1]
        ddi_tbl.putcol("SPECTRAL_WINDOW_ID", spw_ids)
        ddi_tbl.putcol("POLARIZATION_ID", pol_ids)
        ddi_tbl.putcol("FLAG_ROW", np.broadcast_to(False, nrows))


def gen_subt_spw(mspath: str, spw_descr: dict):
    """
    Populates SPECTRAL_WINDOW
    """

    nspws = len(spw_descr)
    with tables.table(
        mspath + "::SPECTRAL_WINDOW", ack=False, readonly=False
    ) as spw_tbl:
        spw_tbl.addrows(nspws)
        # Not in MSv2
        # spw_tbl.putcol("SPECTRAL_WINDOW_ID", list(spw_descr.keys()))
        names = [f"unspecified_test#{idx}" for idx in range(nspws)]
        spw_tbl.putcol("NAME", names)
        # spw_tbl.putcol("REF_FREQUENCY", nspws*[0.9e9])
        # spw_tbl.putcol("TOTAL_BANDWIDTH", nspws*[0.020])

        for spw in range(nspws):
            # nchan = spw_descr[spw]['NUM_CHAN']
            # spw_tbl.addrows(1)  # 1
            nchan = 32

            # spw_tbl.putcell("NAME", spw, "unspecified_test")
            spw_tbl.putcell("REF_FREQUENCY", spw, 0.9e9)

            spw_tbl.putcol("NUM_CHAN", nchan, startrow=spw, nrow=1)
            widths = np.full(nchan, 20000.0)
            spw_tbl.putcell("CHAN_WIDTH", spw, widths)
            # or alternatively
            # spw_tbl.putcol("CHAN_WIDTH", np.full((1, nchan), 20000.0), startrow=spw, nrow=1)
            spw_tbl.putcell("CHAN_FREQ", spw, np.full(nchan, 10e0))
            spw_tbl.putcell("EFFECTIVE_BW", spw, widths)
            spw_tbl.putcell("TOTAL_BANDWIDTH", spw, sum(widths))
            spw_tbl.putcell("RESOLUTION", spw, widths)


def gen_subt_pol_setup(mspath: str, npols, pol_setup_descr: dict):
    """
    populates POLARIZATION
    """

    nsetups = len(pol_setup_descr)

    with tables.table(mspath + "::POLARIZATION", ack=False, readonly=False) as pol_tbl:
        pol_tbl.addrows(npols)
        # pol_tbl.putcol("POLARIZATION_ID", list(spw_descr.keys()))
        pol_tbl.putcol("NUM_CORR", nsetups * [npols])
        pol_tbl.putcol("FLAG_ROW", nsetups * [False])
        corr_types = [9, 11, 13, 15]
        pol_tbl.putcol(
            "CORR_TYPE", np.broadcast_to(corr_types[:npols], (nsetups, npols))
        )
        corr_products = [0, 0, 1, 1]
        pol_tbl.putcol(
            "CORR_PRODUCT", np.broadcast_to(corr_products[:npols], (nsetups, 2, npols))
        )


def gen_subt_antenna(mspath: str, ant_descr: dict):
    """
    create ANTENNA
    """

    with tables.table(mspath + "::ANTENNA", ack=False, readonly=False) as ant_tbl:
        nants = len(ant_descr)
        ant_tbl.addrows(nants)
        ant_tbl.putcol("NAME", [f"test_antenna_{idx}" for idx in range(nants)])
        ant_tbl.putcol("STATION", [f"test_station_{idx}" for idx in range(nants)])
        ant_tbl.putcol("DISH_DIAMETER", nants * [12])
        ant_tbl.putcol("FLAG_ROW", nants * [False])
        ant_tbl.putcol("POSITION", np.broadcast_to([0.01, 0.02, 0.03], (nants, 3)))
        ant_tbl.putcol("POSITION", np.broadcast_to([0.0, 0.0, 0.0], (nants, 3)))


# field and state very relevant for partitions/sub-MSv2
def gen_subt_field(
    mspath: str, fields_descr: dict, gen_ephem: bool = True, misbehave: bool = False
):
    """
    creates FIELD

    only supports polynomials with 1 coefficient

    Parameters
    ----------
    mspath : str
        path of output MS
    fields_descr : dict
        fields description
    gen_ephem : bool
        whether to generate ephemeris fields (with EPHEM pseudo-sub tables
    misbehave : bool
        if True, some missing SOURCE_IDs will be added

    Returns
    -------

    """

    with tables.table(mspath + "::FIELD", ack=False, readonly=False) as fld_tbl:
        nfields = len(fields_descr)
        fld_tbl.addrows(nfields)
        fld_tbl.putcol("NAME", nfields * ["NGC3031"])

        npoly = 0
        fld_tbl.putcol("NUM_POLY", nfields * [npoly])

        adir = np.deg2rad([30, 45])
        for dir_col in ["DELAY_DIR", "PHASE_DIR", "REFERENCE_DIR"]:
            fld_tbl.putcol(dir_col, np.broadcast_to(adir, (nfields, npoly + 1, 2)))

        fld_tbl.putcol("SOURCE_ID", np.arange(nfields))
        if misbehave:
            fld_tbl.putcell("SOURCE_ID", nfields - 1, nfields + 3)

        if gen_ephem:
            tabdesc_ephemeris_id = {
                "EPHEMERIS_ID": {
                    "valueType": "int",
                    "dataManagerType": "StandardStMan",
                    "dataManagerGroup": "StandardStMan",
                    "option": 0,
                    "maxlen": 0,
                    "ndim": 0,
                    "comment": "comment...",
                }
            }
            fld_tbl.addcols(tabdesc_ephemeris_id)
            fld_tbl.putcol("EPHEMERIS_ID", np.broadcast_to(0, (nfields, 1)))

    if gen_ephem:
        gen_subt_ephem(mspath)


def gen_subt_ephem(mspath: str):
    """
    Creates a phony ephemerides table under the FIELD subtable.
    The usual tables... context or 'default_ms_subtable' not working. Needs
    manual coldesc /not in MSv2/3
    """
    # with open_opt_subtable(mspath, "FIELD/EPHEM0_FIELD.tab") as wtbl:
    ephem0_path = Path(mspath) / "FIELD" / "EPHEM0_FIELDNAME.tab"
    tabdesc = {
        "MJD": {
            "valueType": "double",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "keywords": {
                "QuantumUnits": ["s"],
                "MEASINFO": {"type": "epoch", "Ref": "bogus MJD"},
            },
        },
        "RA": {
            "valueType": "double",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "keywords": {
                "UNIT": "deg",
            },
        },
    }

    with tables.table(
        str(ephem0_path), tabledesc=tabdesc, nrow=1, readonly=False, ack=False
    ) as tbl:
        tbl.putcol("MJD", 50000)
        tbl.putcol("RA", 230.334)


def gen_subt_state(mspath: str, states_descr: dict):
    """
    populates STATE
    """

    # intents in OBS_MODE strings
    with tables.table(mspath + "::STATE", ack=False, readonly=False) as st_tbl:
        nstates = len(states_descr)
        st_tbl.addrows(nstates)
        st_tbl.putcol("SIG", nstates * [True])
        st_tbl.putcol("REF", nstates * [False])
        st_tbl.putcol("CAL", nstates * [0.0])
        st_tbl.putcol("LOAD", nstates * [0.0])
        # TODO: generate subscan ids for potentially multiple scans
        st_tbl.putcol("SUB_SCAN", 1 + np.arange(nstates))
        st_tbl.putcol("OBS_MODE", nstates * ["scan_intent#subscan_intent"])
        st_tbl.putcol("FLAG_ROW", nstates * [False])


# Other - optional subtables


@contextmanager
def open_opt_subtable(
    mspath: str, tbl_name: str
) -> Generator[tables.table, None, None]:
    """
    Opens an (optional) subtable of an MS. This can open tables not included
    in the default_ms definition

    Parameters
    ----------
    mspath : str
        path of output MS
    tbl_name : str
        name of the subtable (WEATHER, etc.). Requires a known optional MS
        subtable name that has a known table description.

    Returns
    -------
    Generator[tables.table, None, None]
        context for an optional subtable created as per MSv2 specs
    """
    subt_desc = tables.complete_ms_desc(tbl_name)
    # table = tables.table(mspath + "/" + tbl_name, tabledesc=subt_desc,
    #                dminfo=makedminfo(subt_desc), ack=False, readonly=False)
    table = default_ms_subtable(tbl_name, mspath + "/" + tbl_name, subt_desc)
    try:
        yield table
    finally:
        table.close()


def gen_subt_source(mspath: str, src_descr: dict, misbehave: bool):
    """
    Populate SOURCE subtable, with time dependent source info

    Parameters
    ----------
    mspath : str
        path of output MS
    src_descr : dict
        sources description

    Returns
    -------

    """

    # SOURCE is not included in default_ms
    # with tables.table(mspath + "::SOURCE", ack=False, readonly=False) as src_tbl:
    with open_opt_subtable(mspath, "SOURCE") as src_tbl:
        nsrcs = len(src_descr)
        src_tbl.addrows(nsrcs)
        src_tbl.putcol("SOURCE_ID", list(src_descr.values()))
        src_tbl.putcol("TIME", np.broadcast_to(0, (nsrcs)))
        src_tbl.putcol("INTERVAL", np.broadcast_to(0, (nsrcs)))
        src_tbl.putcol("SPECTRAL_WINDOW_ID", np.broadcast_to(0, (nsrcs)))

        src_tbl.putcol("NAME", np.broadcast_to("test_source_name", (nsrcs)))
        src_tbl.putcol("CALIBRATION_GROUP", np.broadcast_to(0, (nsrcs)))
        src_tbl.putcol("CODE", np.broadcast_to("test_source", (nsrcs)))
        adir = np.deg2rad([29, 34])
        src_tbl.putcol("DIRECTION", np.broadcast_to(adir, (nsrcs, 2)))
        src_tbl.putcol("PROPER_MOTION", np.broadcast_to(adir / 100.0, (nsrcs, 2)))
        # SOURCE_MODEL is optional and its type is TableRecord!
        src_tbl.removecols(["SOURCE_MODEL"])

    with tables.table(mspath, ack=False, readonly=False) as main:
        main.putkeyword("SOURCE", f"Table: {mspath}/SOURCE")


def gen_subt_pointing(mspath: str):
    """
    Populate POINTING subtable, with antenna-based pointing info

    Very rudimentary.
    """
    # with open_opt_subtable(mspath, "POINTING") as tbl:
    nrows = 1
    with tables.table(mspath + "::POINTING", ack=False, readonly=False) as tbl:
        tbl.addrows(nrows)
        tbl.putcol("ANTENNA_ID", 0)
        tbl.putcol("TIME", 2e9)
        tbl.putcol("INTERVAL", 1e12)
        tbl.putcol("NAME", "test_pointing_name")
        tbl.putcol("NUM_POLY", 0)
        tbl.putcol("TIME_ORIGIN", 1e9)
        adir = np.deg2rad([28.98, 34.03])
        tbl.putcol("DIRECTION", np.broadcast_to(adir, (nrows, 1, 2)))
        tbl.putcol("TARGET", np.broadcast_to(adir + 0.01, (nrows, 1, 2)))
        tbl.putcol("TRACKING", True)


# Other, more secondary subtables


def gen_subt_feed(
    mspath: str,
    feed_descr: dict,
    ant_descr: str,
    spw_descr: str,
    misbehave: bool = False,
):
    """
    Populate FEED subtable, with antenna-based pointing info.

    The table is filled for a single FEED_ID (0), for all the available
    antenna IDs and spectral window IDs. The columns are filled with
    roughtly similar values as seen in example ALMA MSs imported with
    importasdm. BEAM_ID is left as -1 (optional BEAM subtable of MSv2 never
    defined).

    Parameters
    ----------
    mspath : str
        path of output MS
    feed_descr : dict
        feed description, only one feed supported
    ant_descr : str

    spw_descr : str

    misbehave : bool
        if misbehave, the FEED table will be empty.

    Returns
    -------

    """
    if misbehave:
        return

    with tables.table(mspath + "::FEED", ack=False, readonly=False) as tbl:
        nfeeds = len(feed_descr)
        nspws = len(spw_descr)
        nants = len(ant_descr)
        nrows = nants * nspws * nfeeds
        tbl.addrows(nrows)
        tbl.putcol("ANTENNA_ID", np.tile(np.arange(nants), int(nrows / nants)))
        tbl.putcol("FEED_ID", np.repeat(0, (nrows)))
        tbl.putcol(
            "SPECTRAL_WINDOW_ID", np.repeat(list(spw_descr.values()), (nrows / nspws))
        )
        tbl.putcol("TIME", np.broadcast_to(0, (nrows)))
        tbl.putcol("INTERVAL", np.broadcast_to(5e9, (nrows)))
        nrecep = 2
        tbl.putcol("NUM_RECEPTORS", np.broadcast_to(nrecep, (nrows)))
        tbl.putcol("BEAM_ID", np.broadcast_to(-1, (nrows)))
        boff = np.deg2rad([0.1, 0.3])
        tbl.putcol("BEAM_OFFSET", np.broadcast_to(boff, (nrows, 2, nrecep)))
        pol_types = ["test_X, test_Y"]
        len_pol_types = 2
        tbl.putcol(
            "POLARIZATION_TYPE", np.broadcast_to(pol_types, (nrows, len_pol_types))
        )
        tbl.putcol("POL_RESPONSE", np.broadcast_to(0.0, (nrows, 2, nrecep)))
        tbl.putcol("POSITION", np.broadcast_to([0.0, 0.0, 0.0], (nrows, 3)))
        tbl.putcol("RECEPTOR_ANGLE", np.broadcast_to([1.51, 0.33], (nrows, nrecep)))


def gen_subt_observation(mspath: str, obs_descr: dict):
    """
    Populate the OBSERVATION table, with one row per observation id/number
    (MSv2: the id is implicitly the row number).
    The string columns are filled with values loosely based on example ALMA
    MSs.

    Parameters
    ----------
    mspath : str
        path of output MS
    obs_descr : dict
        obs description, with IDs, only the len is considered

    Returns
    -------

    """

    nobs = len(obs_descr)
    with tables.table(mspath + "::OBSERVATION", ack=False, readonly=False) as tbl:
        tbl.addrows(nobs)
        # no keys, all data cols:
        # "test_telescope" would produce errors for example in listobs:
        # Exception: Telescope test_telescope is not recognized by CASA.
        # (casa6core::MPosition casa6core::MSMetaData::getObservatoryPosition())
        tbl.putcol("TELESCOPE_NAME", np.repeat("ALMA", nobs))
        tbl.putcol("TIME_RANGE", np.broadcast_to([[4.87021e9, 4.87022e9]], (nobs, 2)))
        tbl.putcol("OBSERVER", np.repeat("Dr. test_observer", nobs))
        tbl.putcol(
            "LOG", np.broadcast_to(["test_obs_log_1", "test_obs_log_2"], (nobs, 2))
        )
        tbl.putcol("SCHEDULE_TYPE", np.repeat("test_schedule_type", nobs))
        sched = np.array(
            [
                [
                    f"SchedulingBlock uid://A002/X1ftest/Xde{idx}",
                    f"ExecBlock uid://A002/X1abtest/X123{idx}",
                ]
                for idx in range(nobs)
            ]
        )
        tbl.putcol("SCHEDULE", sched)
        proj = np.array([f"uid://A002/X1ftest/X4ec{idx}" for idx in range(nobs)])
        tbl.putcol("PROJECT", proj)
        tbl.putcol("RELEASE_DATE", np.repeat(0, nobs))


def gen_subt_processor(mspath: str, proc_descr: dict):
    """
    Populate the PROCESSOR table, with one row per processor id/number
    (MSv2: the id is implicitly the row number). The TYPE_ID is left to -1.
    All values of TYPE are CORRELATOR.
    MODE_ID are also left to -1 (no ..._MODE subtable).
    The string columns are filled with values loosely based on example ALMA
    MSs.

    Parameters
    ----------
    mspath : str
        path of output MS
    proc_descr : dict
        processors description, with IDs, only the len is considered

    Returns
    -------

    """

    nproc = len(proc_descr)
    with tables.table(mspath + "::PROCESSOR", ack=False, readonly=False) as tbl:
        tbl.addrows(nproc)
        # no keys, all data cols:
        # There could also be RADIOMETER, SPECTROMETER, PULSAR-TIMER, etc. - not nor now
        tbl.putcol("TYPE", np.repeat("CORRELATOR", nproc))
        tbl.putcol("SUB_TYPE", np.repeat("test_CORRELATOR_MODE", nproc))
        tbl.putcol("TYPE_ID", np.repeat(-1, nproc))
        tbl.putcol("MODE_ID", np.repeat(-1, nproc))
        tbl.putcol("FLAG_ROW", np.repeat(False, nproc))


def gen_subt_flag_cmd(mspath: str):
    """
    Leaves the FLAG_CMD subtable empty for now, and checks that there are no
    rows.
    """
    with tables.table(mspath + "::FLAG_CMD", ack=False, readonly=False) as tbl:
        assert tbl.nrows() == 0


def gen_subt_history(mspath: str, obs_descr: dict):
    """
    Populate the HISTORY table, with only one row per observation
    The string columns are filled with values loosely based on example ALMA MSs.
    OBJECT_ID is left all to -1.

    Parameters
    ----------
    mspath : str
        path of output MS
    obs_descr : dict
        obs description, with IDs, only the len is considered

    Returns
    -------

    """

    nobs = len(obs_descr)
    with tables.table(mspath + "::HISTORY", ack=False, readonly=False) as tbl:
        tbl.addrows(nobs)
        # keys
        tbl.putcol("TIME", np.repeat(0, nobs))
        tbl.putcol("OBSERVATION_ID", np.arange(nobs))
        # data
        tbl.putcol("MESSAGE", np.repeat("made with test ms generator", nobs))
        tbl.putcol("PRIORITY", np.repeat("INFO", nobs))
        tbl.putcol("ORIGIN", np.repeat("ms_maker", nobs))
        tbl.putcol("OBJECT_ID", np.repeat(0, nobs))
        tbl.putcol("APPLICATION", np.repeat("xradio", nobs))
        tbl.putcol("CLI_COMMAND", np.broadcast_to("gen_subt_history", (nobs, 1)))
        tbl.putcol("APP_PARAMS", np.broadcast_to("", (nobs, 1)))


def gen_subt_syscal(mspath: str, ant_descr: dict):
    """
    Creates a SYSCAL subtable and populates it with a (very incomplete) row
    This is just to enable minimal coverage of some WEATHER handling code in
    the casacore tables read/write functions.
    """
    ncal = len(ant_descr)
    subt_name = "SYSCAL"
    with open_opt_subtable(mspath, subt_name) as sctbl:
        sctbl.addrows(ncal)
        sctbl.putcol("ANTENNA_ID", np.arange(0, ncal))
        sctbl.putcol("FEED_ID", np.repeat(0, ncal))
        sctbl.putcol("SPECTRAL_WINDOW_ID", np.repeat(0, ncal))
        sctbl.putcol("TIME", np.repeat(1e10, ncal))
        sctbl.putcol("INTERVAL", np.repeat(1e12, ncal))
        # all data/flags columns in the SYSCAL table are optional!
        # sctbl.putcol("PHASE_DIFF", np.repeat(0.3, ncal))
        sctbl.putcol("PHASE_DIFF", np.repeat(0.3, ncal))
        # sctbl.putcol("TCAL", np.broadcast_to(50.3, (ncal, 2)))
        sctbl.putcol("TCAL_SPECTRUM", np.broadcast_to(50.3, (ncal, 10, 2)))
        # TRX, etc.

    with tables.table(mspath, ack=False, readonly=False) as main:
        main.putkeyword(subt_name, f"Table: {mspath}/{subt_name}")


def gen_subt_weather(mspath: str):
    """
    Creates a WEATHER subtable and populates it with a (very incomplete) row
    simply with a very high TIME value, as seen in some corner cases of
    test MSs.
    This is just to enable minimal coverage of some WEATHER handling code in
    the casacore tables read/write functions.
    """
    subt_name = "WEATHER"
    with open_opt_subtable(mspath, subt_name) as wtbl:
        wtbl.addrows(1)
        wtbl.putcol("ANTENNA_ID", 0)
        wtbl.putcol("TIME", 1e12)
        wtbl.putcol("INTERVAL", 1e12)
        # all data/flags columns in the WEATHER table are optional!

    with tables.table(mspath, ack=False, readonly=False) as main:
        main.putkeyword(subt_name, f"Table: {mspath}/{subt_name}")


def gen_subt_asdm_receiver(mspath: str):
    """
    **** Produces an empty table, for basic coverage of ASDM_* subtables handling ****
    code.
    Simply creates an empty table and checks no rwos
    """
    subt_name = "ASDM_RECEIVER"
    rec_path = Path(mspath) / subt_name
    tabdesc = {
        "receiverId": {
            "valueType": "int",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "keywords": {},
        },
        "spectralWindowId": {
            "valueType": "int",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "keywords": {},
        },
        "timeInterval": {
            "valueType": "double",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "keywords": {},
        },
    }

    with tables.table(
        str(rec_path), tabledesc=tabdesc, nrow=1, readonly=False, ack=False
    ) as tbl:
        tbl.putcol("receiverId", 0)
        tbl.putcol("spectralWindowId", 0)
        tbl.putcol("timeInterval", 0)

    with tables.table(mspath, ack=False, readonly=False) as main:
        main.putkeyword(subt_name, f"Table: {mspath}/{subt_name}")


def gen_subt_asdm_execblock(mspath: str, misbehave: bool = False):
    """
    Produces a basic ASDM/EXECBLOCK table, for basic coverage of code that handles the ASDM_*
    subtables.
    For now it simply creates a table with one row
    """
    subt_name = "ASDM_EXECBLOCK"
    rec_path = Path(mspath) / subt_name
    tabdesc = {
        "execBlockIDId": {
            "valueType": "string",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "keywords": {},
        },
        "execBlockNumId": {
            "valueType": "string",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "keywords": {},
        },
        "execBlockUID": {
            "valueType": "string",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "keywords": {},
        },
        "sessionReference": {
            "valueType": "string",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "keywords": {},
        },
        "observingScript": {
            "valueType": "string",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "keywords": {},
        },
        "observingScriptUID": {
            "valueType": "string",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "keywords": {},
        },
        "observingLog": {
            "valueType": "string",
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "StandardStMan",
            "option": 0,
            "maxlen": 0,
            "comment": "comment...",
            "ndim": 1,
            "_c_order": True,
            "keywords": {},
        },
    }

    if misbehave:
        nrows = 0
    else:
        nrows = 1
    with tables.table(
        str(rec_path), tabledesc=tabdesc, nrow=nrows, readonly=False, ack=False
    ) as tbl:
        tbl.putcol("execBlockIDId", "1")
        tbl.putcol("execBlockNumId", "3")
        tbl.putcol("execBlockUID", "uid://A001/X1abtest/X123")
        tbl.putcol("sessionReference", "test_session_ref")
        tbl.putcol("observingScript", "test script")
        tbl.putcol("observingScriptUID", "uid://A003/X1abtest/X987")
        tbl.putcol("observingLog", np.broadcast_to("test log line 0", (nrows, 1)))
        # tbl.putcol("observingLog", np.broadcast_to(np.array(["test log line 0", "test log line 1"], dtype="str"), (nrows, 2)))

    with tables.table(mspath, ack=False, readonly=False) as main:
        main.putkeyword(subt_name, f"Table: {mspath}/{subt_name}")
