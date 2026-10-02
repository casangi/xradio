"""
Synthetic ALMA ASDM writer for end-to-end tests of the ``xradio_asdm`` backend.

This is a pytest-free helper module. It writes a REAL on-disk ASDM directory:

- the ASDM metadata tables are pyasdm tables populated from XML rows (like the
  builders in ``conftest.py``) and written with ``pyasdm.ASDM.toFile``;
- one real MIME BDF per Main row is written under ``<dir>/ASDMBinary/`` at the
  path that ``pyasdm.MainRow.getBDFPath()`` resolves to, and the BDFs are
  parseable by ``pyasdm.bdf.BDFReader``.

Every value written to the BDFs encodes its own position, so that the expected
(true) arrays can be computed independently of the xradio code:

- cross-correlations: ``re = cross_re_code(baseline, channel, polarization)``
  and ``im = cross_im_code(integration_code, spw)``, i.e. the (re, im) pair is
  unique for every (integration, baseline, spw, channel, polarization). Integer
  BDF data store the codes, the true visibility is ``code / scaleFactor``.
- auto-correlations (always float32): ``auto_value(code, antenna, spw, channel,
  k)`` where ``k`` is the index of the float within the channel. For
  full-polarization data (``sdPolProducts = XX XY YY``) the four floats per
  channel are ``XX, Re(XY), Im(XY), YY``.
- flags: int32 bit words, see ``flag_word``. Several distinct non-zero bits
  (``1``, ``16``, ``2**30``, ``-2**31`` and combinations) are used, and the
  expected (boolean) flag is ``word != 0`` (contract K2).

``integration_code`` is the running index of an integration (or of a TIM
sample, for packed BDFs) within one ConfigDescription, in time order.

Main entry points
-----------------
- ``write_synthetic_asdm(spec, directory)`` writes an ASDM described by an
  ``ASDMSpec`` and returns a ``SyntheticASDM`` (the "truth").
- The ``*_spec()`` factories give canonical layouts (interferometric dual-pol
  multi-spw, full-pol, single-dish, mosaic, interleaved configurations, ...).
- ``SyntheticASDM.expected_partitions()`` computes, independently of xradio,
  the partitions expected per contract K7; ``match_partition`` maps an opened
  MSv4 node to one of them.
- ``expected_visibility``, ``expected_spectrum``, ``expected_flags``,
  ``expected_times`` compute the true arrays for a list of integrations.
- ``verify_bdf_roundtrip`` reads every BDF back with pyasdm's ``BDFReader``
  and checks the raw arrays.
"""

from __future__ import annotations

import contextlib
import datetime
import io
import os
import re
from dataclasses import dataclass

import numpy as np
import pyasdm

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: seconds between the ASDM/MJD epoch (1858-11-17) and the unix epoch (1970-01-01)
ASDM_TO_UNIX_OFFSET_S = 3_506_716_800
ASDM_TO_UNIX_OFFSET_NS = ASDM_TO_UNIX_OFFSET_S * 10**9
#: 2023-07-22T04:26:40 UTC: inside the IERS-B table bundled with astropy, so
#: UVW calculations never need to download anything.
DEFAULT_T0_UNIX_S = 1_690_000_000
DEFAULT_T0_NS = (DEFAULT_T0_UNIX_S + ASDM_TO_UNIX_OFFSET_S) * 10**9

#: ITRF position used as array center (ALMA site, as in calculate_uvw)
ARRAY_CENTER_ITRF = np.array(
    [2225142.180268967, -5440307.370348562, -2481029.851873547]
)
#: station offsets (ITRF meters) w.r.t. ARRAY_CENTER_ITRF. Baselines up to ~1.3 km
#: make UVW sensitive to time errors of a few seconds.
STATION_OFFSETS_ITRF = np.array(
    [
        [0.0, 0.0, 0.0],
        [312.25, 141.5, -96.75],
        [-205.5, 498.0, 251.25],
        [790.0, -310.5, 402.0],
        [-611.0, -695.25, 120.5],
        [150.0, 820.0, -330.0],
    ]
)

#: Flag words used by ``flag_word`` (int32). Zero entries mean "not flagged".
FLAG_WORDS = np.array(
    [0, 1, 0, 16, 0, 2**30, 0, -(2**31), 0, 0, 1 | 16 | 2**30, 0], dtype=np.int32
)

CROSS_NP_TYPES = {
    "FLOAT32_TYPE": np.dtype("<f4"),
    "INT32_TYPE": np.dtype("<i4"),
    "INT16_TYPE": np.dtype("<i2"),
}

_SPECTRAL_TYPE_SHORT = {
    "FULL_RESOLUTION": "FULL_RES",
    "CHANNEL_AVERAGE": "CH_AVG",
    "BASEBAND_WIDE": "SQLD",
}

_SUBSCAN_INTENT_STATE = {
    # subscan intent -> (calDeviceName, sig, ref, onSky)
    "ON_SOURCE": ("NONE", "true", "false", "true"),
    "OFF_SOURCE": ("NONE", "false", "true", "true"),
    "HOT": ("HOT_LOAD", "false", "false", "false"),
    "AMBIENT": ("AMBIENT_LOAD", "true", "false", "false"),
    "SCANNING": ("NONE", "true", "true", "true"),
}


# --------------------------------------------------------------------------
# Specification of a synthetic ASDM
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PolSetup:
    """Polarization products of one SPW, as written in the BDF header."""

    cross: tuple[str, ...]
    sd: tuple[str, ...]

    @property
    def num_auto_values(self) -> int:
        """Number of float32 values per channel in autoData."""
        return 4 if len(self.sd) == 3 else len(self.sd)


POL_SETUPS = {
    "single": PolSetup(("XX",), ("XX",)),
    "dual": PolSetup(("XX", "YY"), ("XX", "YY")),
    "full": PolSetup(("XX", "XY", "YX", "YY"), ("XX", "XY", "YY")),
    "stokes_i": PolSetup((), ("I",)),
}


@dataclass
class SpwSpec:
    """One spectral window of a baseband."""

    nchan: int
    pol: str = "dual"
    #: frequency of channel 0 (center). None: assigned automatically (unique).
    chan_freq_start: float | None = None
    chan_freq_step: float = 15.625e6
    sideband: str = "USB"
    #: write chanFreqArray instead of chanFreqStart/chanFreqStep
    use_chan_freq_array: bool = False
    num_bin: int = 1


@dataclass
class BasebandSpec:
    name: str
    spws: list[SpwSpec]


@dataclass
class ConfigSpec:
    """A ConfigDescription (and the BDFs written for it)."""

    basebands: list[BasebandSpec]
    correlation_mode: str = "CROSS_AND_AUTO"
    processor_type: str = "CORRELATOR"
    spectral_type: str = "FULL_RESOLUTION"
    cross_type: str = "FLOAT32_TYPE"
    scale_factor: float = 1.0
    apc: tuple[str, ...] = ("AP_UNCORRECTED",)
    #: > 0: packed BDF (dimensionality 0, numTime=N, a single subset per BDF)
    packed_num_time: int = 0
    #: number of integrations per subscan (None: use SubscanSpec.num_integrations)
    num_integrations: int | None = None
    with_actual_times: bool = False
    with_flags: bool = True
    #: explicit DataDescription ids of this config, in config order (the SPW id
    #: is then equal to the DD id). None: assigned sequentially.
    dd_ids: list[int] | None = None

    @property
    def spws(self) -> list[SpwSpec]:
        return [spw for bb in self.basebands for spw in bb.spws]

    @property
    def is_auto_only(self) -> bool:
        return self.correlation_mode == "AUTO_ONLY"

    @property
    def loaded_apc_index(self) -> int:
        """
        Index (in ``apc``) of the APC whose crossData the backend loads (K6):
        AP_UNCORRECTED, or the only APC when there is one (whatever it is).
        """
        if len(self.apc) <= 1:
            return 0
        return list(self.apc).index("AP_UNCORRECTED")


def apc_code_offset(apc_name: str) -> int:
    """APC code offset the writer encodes the crossData of an APC with: 0 for
    AP_UNCORRECTED, 1 for any other APC."""
    return 0 if apc_name == "AP_UNCORRECTED" else 1


@dataclass
class FieldSpec:
    name: str
    #: (ra, dec) in rad, ICRS
    phase_dir: tuple[float, float]
    reference_dir: tuple[float, float] | None = None
    delay_dir: tuple[float, float] | None = None
    source_name: str | None = None
    direction_code: str = "ICRS"


@dataclass
class SubscanSpec:
    field_id: int = 0
    intent: str = "ON_SOURCE"
    num_integrations: int = 2


@dataclass
class ScanSpec:
    subscans: list[SubscanSpec]
    intents: tuple[str, ...] = ("OBSERVE_TARGET",)


@dataclass
class ASDMSpec:
    configs: list[ConfigSpec]
    scans: list[ScanSpec]
    fields: list[FieldSpec]
    num_antenna: int = 3
    name: str = "uid___A002_X1234_X5678"
    t0_ns: int = DEFAULT_T0_NS
    integration_ns: int = 1_152_000_000
    subscan_gap_ns: int = 2_000_000_000
    scan_gap_ns: int = 10_000_000_000
    with_pointing: bool = False
    pointing_num_sample: int = 4
    #: antennas (indices) that have Pointing rows, None: all. The truth has NaN
    #: pointing values for the other antennas.
    pointing_antennas: tuple[int, ...] | None = None
    #: write the Pointing table in binary form (pyasdm default). XML is the
    #: default here to not depend on the fix of pyasdm ArrayTimeInterval.fromBin
    pointing_as_bin: bool = False
    exec_block_num: int = 1
    #: optional ExecBlock.releaseDate (ASDM ArrayTime ns)
    release_date_ns: int | None = None


# --------------------------------------------------------------------------
# Truth (what was written)
# --------------------------------------------------------------------------


@dataclass
class SpwTruth:
    spw_id: int
    dd_id: int
    config_idx: int
    #: position of the SPW in the config's DD list == position in the BDF
    #: (baseband major, spw minor)
    bdf_index: int
    baseband_name: str
    baseband_index: int
    index_in_baseband: int
    nchan: int
    pol: PolSetup
    num_bin: int
    chan_freqs: np.ndarray
    correlation_mode: str
    polarization_products: tuple[str, ...]


@dataclass
class BDFTruth:
    path: str
    main_row: int
    config_idx: int
    scan_number: int
    subscan_number: int
    field_id: int
    scan_intents: tuple[str, ...]
    subscan_intent: str
    #: ASDM ArrayTime (ns) of the Main row (midpoint of the subscan)
    main_time_ns: int
    #: midpoint (ns, ASDM ArrayTime) of every integration / TIM sample
    mid_ns: np.ndarray
    #: duration (ns) of every integration / TIM sample
    interval_ns: np.ndarray
    #: value code of every integration / TIM sample
    codes: np.ndarray
    packed: bool

    @property
    def num_times(self) -> int:
        return len(self.mid_ns)

    @property
    def obs_modes(self) -> tuple[str, ...]:
        """MSv2-style INTENT#SUBINTENT strings of this BDF, sorted and unique."""
        return tuple(
            sorted({f"{intent}#{self.subscan_intent}" for intent in self.scan_intents})
        )


@dataclass(frozen=True)
class IntegrationRef:
    bdf: BDFTruth
    index: int

    @property
    def mid_ns(self) -> int:
        return int(self.bdf.mid_ns[self.index])

    @property
    def code(self) -> int:
        return int(self.bdf.codes[self.index])


@dataclass
class ExpectedPartition:
    spw_id: int
    dd_id: int
    config_idx: int
    obs_modes: tuple[str, ...]
    field_ids: tuple[int, ...]
    scan_numbers: tuple[int, ...]
    subscan_numbers: tuple[int, ...]
    bdfs: list[BDFTruth]
    integrations: list[IntegrationRef]

    @property
    def unix_times(self) -> np.ndarray:
        return expected_times(self.integrations)


@dataclass
class SyntheticASDM:
    path: str
    spec: ASDMSpec
    spws: list[SpwTruth]
    bdfs: list[BDFTruth]
    antenna_names: list[str]
    station_names: list[str]
    #: ITRF positions (m) of the antennas (pad + zero antenna offsets)
    antenna_positions: np.ndarray
    field_names: list[str]
    source_names: list[str]
    field_source_ids: list[int]
    #: pointing samples: unix time (s) of every sample (same for all antennas)
    pointing_times_unix: np.ndarray | None = None
    #: target (az, alt) per (sample, antenna), NaN for antennas without Pointing
    #: rows (ASDMSpec.pointing_antennas); pointing_encoder likewise
    pointing_target: np.ndarray | None = None
    pointing_encoder: np.ndarray | None = None

    # ---- simple lookups -------------------------------------------------
    @property
    def num_antenna(self) -> int:
        return self.spec.num_antenna

    @property
    def cross_baselines(self) -> list[tuple[int, int]]:
        """Cross baselines (ant_i, ant_j), i < j, in BDF order."""
        return cross_baseline_pairs(self.num_antenna)

    def spw(self, spw_id: int) -> SpwTruth:
        for spw in self.spws:
            if spw.spw_id == spw_id:
                return spw
        raise KeyError(f"No SPW with id {spw_id}")

    def config(self, config_idx: int) -> ConfigSpec:
        return self.spec.configs[config_idx]

    def spw_id_from_frequency(self, frequency: np.ndarray, atol: float = 1.0) -> int:
        """Identify the SPW from the values of a frequency coordinate."""
        frequency = np.asarray(frequency, dtype=float)
        for spw in self.spws:
            if spw.chan_freqs.shape == frequency.shape and np.allclose(
                spw.chan_freqs, frequency, rtol=0, atol=atol
            ):
                return spw.spw_id
        raise AssertionError(
            f"No synthetic SPW matches the frequency coordinate {frequency}. "
            f"SPWs: {[(s.spw_id, s.chan_freqs) for s in self.spws]}"
        )

    def make_field_name(self, field_id: int) -> str:
        """field_name as expected in MSv4 (contract K8)."""
        return f"{self.field_names[field_id]}_{field_id}"

    # ---- selections -----------------------------------------------------
    def integrations(
        self,
        spw_id: int,
        field_ids=None,
        scan_numbers=None,
        subscan_numbers=None,
        obs_modes=None,
    ) -> list[IntegrationRef]:
        """All integrations of the SPW's config matching the selection, time sorted."""
        cfg_idx = self.spw(spw_id).config_idx
        bdfs = [
            bdf
            for bdf in self.bdfs
            if bdf.config_idx == cfg_idx
            and (field_ids is None or bdf.field_id in field_ids)
            and (scan_numbers is None or bdf.scan_number in scan_numbers)
            and (subscan_numbers is None or bdf.subscan_number in subscan_numbers)
            and (obs_modes is None or bdf.obs_modes == tuple(obs_modes))
        ]
        bdfs = sorted(bdfs, key=lambda bdf: bdf.main_time_ns)
        return [
            IntegrationRef(bdf, idx) for bdf in bdfs for idx in range(bdf.num_times)
        ]

    def expected_partitions(
        self,
        partition_scheme: list[str] | None = None,
        include_processor_types: list[str] | None = None,
        include_spectral_resolution_types: list[str] | None = None,
    ) -> list[ExpectedPartition]:
        """
        Partitions expected per contract K7, computed independently of xradio.

        Mandatory axes: execBlockId (only one here), configDescriptionId,
        dataDescriptionId, scanIntent (the set of INTENT#SUBINTENT).

        Parameters
        ----------
        partition_scheme : list[str] | None
            Optional axes, among "fieldId", "scanNumber", "subscanNumber".
            None means ["fieldId"] (the open_asdm default).
        include_processor_types : list[str] | None
            None means ["CORRELATOR", "SPECTROMETER"] (open_asdm default).
        include_spectral_resolution_types : list[str] | None
            None means ["FULL_RESOLUTION", "BASEBAND_WIDE"] (open_asdm default).

        Returns
        -------
        list[ExpectedPartition]
            One entry per expected MSv4, with its BDFs and integrations in
            Main time order.
        """
        if partition_scheme is None:
            partition_scheme = ["fieldId"]
        if include_processor_types is None:
            include_processor_types = ["CORRELATOR", "SPECTROMETER"]
        if include_spectral_resolution_types is None:
            include_spectral_resolution_types = ["FULL_RESOLUTION", "BASEBAND_WIDE"]
        axis_attr = {
            "fieldId": "field_id",
            "scanNumber": "scan_number",
            "subscanNumber": "subscan_number",
        }
        groups: dict[tuple, list[BDFTruth]] = {}
        for bdf in sorted(self.bdfs, key=lambda bdf: (bdf.main_time_ns, bdf.main_row)):
            cfg = self.config(bdf.config_idx)
            if cfg.processor_type not in include_processor_types:
                continue
            if cfg.spectral_type not in include_spectral_resolution_types:
                continue
            for spw in self.spws:
                if spw.config_idx != bdf.config_idx:
                    continue
                key = (spw.spw_id, bdf.obs_modes) + tuple(
                    getattr(bdf, axis_attr[axis]) for axis in partition_scheme
                )
                groups.setdefault(key, []).append(bdf)

        partitions = []
        for key, bdfs in groups.items():
            spw = self.spw(key[0])
            partitions.append(
                ExpectedPartition(
                    spw_id=spw.spw_id,
                    dd_id=spw.dd_id,
                    config_idx=spw.config_idx,
                    obs_modes=key[1],
                    field_ids=tuple(sorted({bdf.field_id for bdf in bdfs})),
                    scan_numbers=tuple(sorted({bdf.scan_number for bdf in bdfs})),
                    subscan_numbers=tuple(sorted({bdf.subscan_number for bdf in bdfs})),
                    bdfs=bdfs,
                    integrations=[
                        IntegrationRef(bdf, idx)
                        for bdf in bdfs
                        for idx in range(bdf.num_times)
                    ],
                )
            )
        return partitions


# --------------------------------------------------------------------------
# Value encoding (the ground truth)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeLayout:
    """Radices used to encode positions into values, per config."""

    max_nchan: int
    num_spw: int
    num_antenna: int
    config_idx: int


def cross_baseline_pairs(num_antenna: int) -> list[tuple[int, int]]:
    """Cross baselines (i, j), i < j, in BDF order: (0,1), (0,2), (1,2), (0,3), ..."""
    return [(i, j) for j in range(num_antenna) for i in range(j)]


def cross_re_code(layout: CodeLayout, bl, chan, pol, apc_idx=0, bin_idx=0):
    """Real part code of crossData (unique per baseline, channel, polarization)."""
    return (
        (
            1
            + np.asarray(pol)
            + 4 * (np.asarray(chan) + layout.max_nchan * np.asarray(bl))
        )
        + 1000 * np.asarray(apc_idx)
        + 2000 * np.asarray(bin_idx)
    )


def cross_im_code(layout: CodeLayout, code, ospw):
    """Imaginary part code of crossData (unique per integration, SPW and config)."""
    return -(
        1
        + np.asarray(ospw)
        + layout.num_spw * np.asarray(code)
        + 500 * layout.config_idx
    )


def auto_value(layout: CodeLayout, code, ant, ospw, chan, k, bin_idx=0):
    """float value of the k-th float of channel `chan` of antenna `ant` (exact in float32)."""
    return (
        0.5
        + np.asarray(k)
        + 4.0
        * (
            np.asarray(chan)
            + layout.max_nchan
            * (
                np.asarray(ant)
                + layout.num_antenna
                * (np.asarray(ospw) + layout.num_spw * np.asarray(code))
            )
        )
        + 1.0e5 * layout.config_idx
        + 5.0e4 * np.asarray(bin_idx)
    )


def flag_word(layout: CodeLayout, code, row, ospw, k):
    """int32 BDF flag word. row: cross baseline index, or nbl + antenna for autos."""
    idx = (
        5 * np.asarray(code)
        + 7 * np.asarray(row)
        + 3 * np.asarray(ospw)
        + 11 * np.asarray(k)
        + 13 * layout.config_idx
    ) % len(FLAG_WORDS)
    return FLAG_WORDS[idx]


def code_layout(spec: ASDMSpec, config_idx: int) -> CodeLayout:
    """Value-encoding radices of one config."""
    cfg = spec.configs[config_idx]
    return CodeLayout(
        max_nchan=max(spw.nchan for spw in cfg.spws),
        num_spw=len(cfg.spws),
        num_antenna=spec.num_antenna,
        config_idx=config_idx,
    )


# --------------------------------------------------------------------------
# Expected arrays
# --------------------------------------------------------------------------


def expected_times(integrations: list[IntegrationRef]) -> np.ndarray:
    """
    Expected time values of integrations (contract K1).

    Parameters
    ----------
    integrations : list[IntegrationRef]
        Integrations, e.g. from ``SyntheticASDM.integrations`` or
        ``ExpectedPartition.integrations``.

    Returns
    -------
    np.ndarray
        float64 seconds since 1970-01-01 (``(ns - 3506716800e9) / 1e9``).
    """
    ns = np.array([ref.mid_ns for ref in integrations], dtype=np.int64)
    return (ns - ASDM_TO_UNIX_OFFSET_NS) / 1e9


def expected_datetimes(integrations: list[IntegrationRef]) -> list[datetime.datetime]:
    """
    True UTC instants (naive datetimes, UTC) of the integrations, computed with
    datetime arithmetic from the ASDM epoch (independently of the unix-seconds
    formula).
    """
    epoch = datetime.datetime(1858, 11, 17)
    return [
        epoch + datetime.timedelta(microseconds=ref.mid_ns / 1000)
        for ref in integrations
    ]


def expected_intervals(integrations: list[IntegrationRef]) -> np.ndarray:
    """Expected durations (s) of the integrations."""
    return (
        np.array(
            [ref.bdf.interval_ns[ref.index] for ref in integrations], dtype=np.int64
        )
        / 1e9
    )


def _auto_values_per_pol(
    truth: SyntheticASDM, spw: SpwTruth, code: int, ant: int, for_spectrum: bool
) -> np.ndarray:
    """(nchan, npol) expected values of one antenna's autocorrelation."""
    layout = code_layout(truth.spec, spw.config_idx)
    chan = np.arange(spw.nchan)[:, None]
    nvals = spw.pol.num_auto_values
    vals = auto_value(layout, code, ant, spw.bdf_index, chan, np.arange(nvals)[None, :])
    if len(spw.pol.sd) != 3:
        return vals.astype(complex) if not for_spectrum else vals
    xx, re_xy, im_xy, yy = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]
    if for_spectrum:
        # real SPECTRUM (K12): real part of XX, XY, YY
        return np.stack([xx, re_xy, yy], axis=-1)
    xy = re_xy + 1j * im_xy
    if len(spw.polarization_products) == 4:
        return np.stack([xx, xy, np.conj(xy), yy], axis=-1)
    return np.stack([xx, xy, yy], axis=-1)


def expected_visibility(
    truth: SyntheticASDM, spw_id: int, integrations: list[IntegrationRef]
) -> np.ndarray:
    """
    Expected correlated data of one SPW.

    Parameters
    ----------
    truth : SyntheticASDM
        Truth returned by ``write_synthetic_asdm``.
    spw_id : int
        SpectralWindow id (== DataDescription id in the synthetic ASDMs).
    integrations : list[IntegrationRef]
        Integrations (of the SPW's config) along the time axis.

    Returns
    -------
    np.ndarray
        complex128 array (time, baseline, frequency, polarization). Baselines are
        the cross baselines in BDF order followed by the autos in antenna order
        (CROSS_AND_AUTO), or the antennas (AUTO_ONLY) (K4). Integer data are
        divided by the scale factor (K3), the AP_UNCORRECTED data are used (K6)
        and full-pol autos are ``[XX, XY, conj(XY), YY]`` (``[XX, XY, YY]`` for
        AUTO_ONLY).
    """
    spw = truth.spw(spw_id)
    cfg = truth.config(spw.config_idx)
    layout = code_layout(truth.spec, spw.config_idx)
    nant = truth.num_antenna
    nbl = 0 if cfg.is_auto_only else len(truth.cross_baselines)
    npol = len(spw.polarization_products)
    out = np.zeros((len(integrations), nbl + nant, spw.nchan, npol), dtype=complex)
    # The backend loads the AP_UNCORRECTED block, or the only APC (K6). The writer
    # encodes AP_UNCORRECTED with APC code offset 0 whatever its position in the
    # APC axis, and any other APC with offset 1.
    apc_offset = apc_code_offset(cfg.apc[cfg.loaded_apc_index])
    chan = np.arange(spw.nchan)[:, None]
    pol = np.arange(npol)[None, :]
    for tidx, ref in enumerate(integrations):
        assert ref.bdf.config_idx == spw.config_idx
        for bl in range(nbl):
            re = cross_re_code(layout, bl, chan, pol, apc_offset)
            im = (
                cross_im_code(layout, ref.code, spw.bdf_index)
                - 1000 * apc_offset
                + 0 * re
            )
            out[tidx, bl] = (re + 1j * im) / cfg.scale_factor
        for ant in range(nant):
            out[tidx, nbl + ant] = _auto_values_per_pol(
                truth, spw, ref.code, ant, for_spectrum=False
            )
    return out


def expected_spectrum(
    truth: SyntheticASDM, spw_id: int, integrations: list[IntegrationRef]
) -> np.ndarray:
    """
    Expected real SPECTRUM of one single-dish SPW (K12).

    Parameters
    ----------
    truth : SyntheticASDM
        Truth returned by ``write_synthetic_asdm``.
    spw_id : int
        SpectralWindow id.
    integrations : list[IntegrationRef]
        Integrations along the time axis.

    Returns
    -------
    np.ndarray
        float64 array (time, antenna_name, frequency, polarization) with the
        real part of the auto-correlations.
    """
    spw = truth.spw(spw_id)
    out = np.zeros(
        (
            len(integrations),
            truth.num_antenna,
            spw.nchan,
            len(spw.polarization_products),
        )
    )
    for tidx, ref in enumerate(integrations):
        for ant in range(truth.num_antenna):
            out[tidx, ant] = _auto_values_per_pol(
                truth, spw, ref.code, ant, for_spectrum=True
            )
    return out


def expected_flags(
    truth: SyntheticASDM, spw_id: int, integrations: list[IntegrationRef]
) -> np.ndarray:
    """
    Expected FLAG of one SPW.

    Parameters
    ----------
    truth : SyntheticASDM
        Truth returned by ``write_synthetic_asdm``.
    spw_id : int
        SpectralWindow id.
    integrations : list[IntegrationRef]
        Integrations along the time axis.

    Returns
    -------
    np.ndarray
        bool array with the dims of ``expected_visibility``: flagged == (BDF
        flag word != 0) (K2), broadcast over frequency. For full-pol autos (3
        sd flag words XX, XY, YY) the YX flag is the XY flag.
    """
    spw = truth.spw(spw_id)
    cfg = truth.config(spw.config_idx)
    layout = code_layout(truth.spec, spw.config_idx)
    nant = truth.num_antenna
    nbl = 0 if cfg.is_auto_only else len(truth.cross_baselines)
    npol = len(spw.polarization_products)
    out = np.zeros((len(integrations), nbl + nant, spw.nchan, npol), dtype=bool)
    if not cfg.with_flags:
        return out
    for tidx, ref in enumerate(integrations):
        for bl in range(nbl):
            words = flag_word(layout, ref.code, bl, spw.bdf_index, np.arange(npol))
            out[tidx, bl] = (words != 0)[None, :]
        nsd = len(spw.pol.sd)
        for ant in range(nant):
            words = flag_word(
                layout, ref.code, nbl + ant, spw.bdf_index, np.arange(nsd)
            )
            flagged = words != 0
            if nsd == 3 and npol == 4:
                flagged = flagged[[0, 1, 1, 2]]
            out[tidx, nbl + ant] = flagged[None, :]
    return out


def match_partition(
    truth: SyntheticASDM, node_ds, partitions: list[ExpectedPartition]
) -> ExpectedPartition:
    """
    Map an opened MSv4 dataset to one of the expected partitions.

    The SPW is identified from the frequency coordinate. If several partitions
    share the SPW they are disambiguated by the time coordinate (exact
    match), then by (number of times, scan_name values, field_name values),
    then by the scan_intents attribute.

    Parameters
    ----------
    truth : SyntheticASDM
        Truth returned by ``write_synthetic_asdm``.
    node_ds : xr.Dataset
        Correlated dataset of an MSv4 node.
    partitions : list[ExpectedPartition]
        Output of ``SyntheticASDM.expected_partitions``.

    Returns
    -------
    ExpectedPartition
        The matching partition. AssertionError is raised if there is no unique
        match.
    """
    spw_id = truth.spw_id_from_frequency(node_ds.frequency.values)
    candidates = [part for part in partitions if part.spw_id == spw_id]
    if len(candidates) == 1:
        return candidates[0]
    times = np.asarray(node_ds.time.values, dtype=float)
    for part in candidates:
        if len(part.integrations) == len(times) and np.allclose(
            part.unix_times, times, rtol=0, atol=1e-3
        ):
            return part
    scans = set(map(str, np.unique(node_ds.scan_name.values)))
    fields = set(map(str, np.unique(node_ds.field_name.values)))
    matches = [
        part
        for part in candidates
        if len(part.integrations) == len(times)
        and {str(num) for num in part.scan_numbers} == scans
        and {truth.make_field_name(fid) for fid in part.field_ids} == fields
    ]
    if len(matches) > 1:
        intents = node_ds.scan_name.attrs.get("scan_intents")
        if isinstance(intents, list):
            matches = [
                part
                for part in matches
                if set(map(str, intents)) == set(part.obs_modes)
            ]
    if len(matches) == 1:
        return matches[0]
    raise AssertionError(
        f"Cannot match node (spw {spw_id}, {len(times)} times, scans {scans}, "
        f"fields {fields}) to one expected partition. Candidates: "
        + str(
            [
                (p.obs_modes, p.field_ids, p.scan_numbers, len(p.integrations))
                for p in candidates
            ]
        )
    )


def baseline_vectors_itrf(truth: SyntheticASDM, auto_only: bool = False) -> np.ndarray:
    """ITRF vectors POSITION(ant1) - POSITION(ant2), (baseline, 3), MSv4 baseline order."""
    pos = truth.antenna_positions
    pairs = [] if auto_only else truth.cross_baselines
    pairs = pairs + [(ant, ant) for ant in range(truth.num_antenna)]
    return np.array([pos[i] - pos[j] for i, j in pairs])


def phase_center_itrs_unit_vectors(
    unix_times: np.ndarray, ra_dec: np.ndarray
) -> np.ndarray:
    """
    Unit vectors (time, 3) towards (ra, dec) [rad, ICRS, one per time] in the
    ITRS frame at the given UTC unix times. Independent reference for w.
    """
    import astropy.units as u
    from astropy.coordinates import ITRS, SkyCoord
    from astropy.time import Time

    ra_dec = np.broadcast_to(np.asarray(ra_dec, dtype=float), (len(unix_times), 2))
    obstime = Time(np.asarray(unix_times, dtype=float), format="unix", scale="utc")
    sky = SkyCoord(ra_dec[:, 0] * u.rad, ra_dec[:, 1] * u.rad, frame="icrs")
    itrs = sky.transform_to(ITRS(obstime=obstime))
    xyz = itrs.cartesian.xyz.value.T
    return xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)


# --------------------------------------------------------------------------
# Raw BDF arrays (layout of the BDF specification)
# --------------------------------------------------------------------------


@dataclass
class _ConfigLayout:
    cfg: ConfigSpec
    spws: list[SpwTruth]
    layout: CodeLayout
    num_antenna: int

    @property
    def nbl(self) -> int:
        if self.cfg.is_auto_only:
            return 0
        return self.num_antenna * (self.num_antenna - 1) // 2


def _raw_cross(cl: _ConfigLayout, code: int) -> np.ndarray:
    """crossData of one integration: BAL BAB SPW [BIN] [APC] SPP POL (re, im)."""
    parts = []
    napc = len(cl.cfg.apc)
    for bl in range(cl.nbl):
        for spw in cl.spws:
            npol = len(spw.pol.cross)
            b, a, c, p = np.meshgrid(
                np.arange(spw.num_bin),
                np.arange(napc),
                np.arange(spw.nchan),
                np.arange(npol),
                indexing="ij",
            )
            apc_offset = np.array([apc_code_offset(name) for name in cl.cfg.apc])[a]
            re = cross_re_code(cl.layout, bl, c, p, apc_offset, b)
            im = cross_im_code(cl.layout, code, spw.bdf_index) - 1000 * apc_offset
            parts.append(np.stack([re, im], axis=-1).ravel())
    return np.concatenate(parts) if parts else np.zeros(0)


def _raw_auto(cl: _ConfigLayout, code: int) -> np.ndarray:
    """autoData of one integration / TIM sample: ANT BAB SPW [BIN] SPP POL."""
    parts = []
    for ant in range(cl.num_antenna):
        for spw in cl.spws:
            b, c, k = np.meshgrid(
                np.arange(spw.num_bin),
                np.arange(spw.nchan),
                np.arange(spw.pol.num_auto_values),
                indexing="ij",
            )
            parts.append(
                auto_value(cl.layout, code, ant, spw.bdf_index, c, k, b).ravel()
            )
    return np.concatenate(parts).astype("<f4")


def _raw_flags(cl: _ConfigLayout, code: int) -> np.ndarray:
    """flags of one integration / TIM sample: [BAL BAB SPW POL] then [ANT BAB SPW POL]."""
    parts = []
    for bl in range(cl.nbl):
        for spw in cl.spws:
            parts.append(
                flag_word(
                    cl.layout, code, bl, spw.bdf_index, np.arange(len(spw.pol.cross))
                )
            )
    for ant in range(cl.num_antenna):
        for spw in cl.spws:
            parts.append(
                flag_word(
                    cl.layout,
                    code,
                    cl.nbl + ant,
                    spw.bdf_index,
                    np.arange(len(spw.pol.sd)),
                )
            )
    return np.concatenate(parts).astype("<i4")


def _cross_to_stored(cl: _ConfigLayout, raw: np.ndarray) -> np.ndarray:
    np_type = CROSS_NP_TYPES[cl.cfg.cross_type]
    if np_type.kind == "i":
        info = np.iinfo(np_type)
        if raw.size and (raw.min() < info.min or raw.max() > info.max):
            raise ValueError(f"Synthetic codes do not fit {cl.cfg.cross_type}")
        return raw.astype(np_type)
    # float data are used as is (K3): store the true value code / scale
    return (raw / cl.cfg.scale_factor).astype(np_type)


def _axes(cl: _ConfigLayout, component: str) -> str:
    tim = "TIM " if cl.cfg.packed_num_time else ""
    has_bin = any(spw.num_bin > 1 for spw in cl.spws)
    binax = "BIN " if has_bin else ""
    if component == "crossData":
        apc = "APC " if len(cl.cfg.apc) > 1 else ""
        return f"BAL BAB SPW {binax}{apc}SPP POL"
    if component == "autoData":
        return f"{tim}ANT BAB SPW {binax}SPP POL"
    if component == "flags":
        return f"{tim}BAL ANT BAB SPW POL" if cl.nbl else f"{tim}ANT BAB SPW POL"
    if component in ("actualTimes", "actualDurations"):
        return f"{tim}ANT"
    raise ValueError(component)


# --------------------------------------------------------------------------
# BDF writer
# --------------------------------------------------------------------------

_B1 = "MIME_boundary-1"
_B2 = "MIME_boundary-2"


def _write_bdf(path: str, cl: _ConfigLayout, bdf: BDFTruth, spec: ASDMSpec) -> None:
    cfg = cl.cfg
    packed = cfg.packed_num_time > 0
    if packed and not cfg.is_auto_only:
        raise ValueError(
            "Packed (dimensionality 0) BDFs are only supported for AUTO_ONLY"
        )
    nant = cl.num_antenna
    eb, scan, subscan = spec.exec_block_num, bdf.scan_number, bdf.subscan_number

    bb_xml = ""
    for bb in cfg.basebands:
        bb_xml += f'<baseband name="{bb.name}">'
        for spw_idx, spw_spec in enumerate(bb.spws):
            pol = POL_SETUPS[spw_spec.pol]
            attrs = (
                f'sw="{spw_idx + 1}" swbb="{bb.name}_SW-{spw_idx + 1}" '
                f'numSpectralPoint="{spw_spec.nchan}" numBin="{spw_spec.num_bin}" '
                f'sideband="{spw_spec.sideband}"'
            )
            if not cfg.is_auto_only:
                attrs += (
                    f' crossPolProducts="{" ".join(pol.cross)}"'
                    f' scaleFactor="{cfg.scale_factor!r}"'
                )
            attrs += f' sdPolProducts="{" ".join(pol.sd)}"'
            bb_xml += f"<spectralWindow {attrs}/>"
        bb_xml += "</baseband>"

    ntim = cfg.packed_num_time if packed else 1
    flags_size = _raw_flags(cl, 0).size * ntim if cfg.with_flags else 0
    auto_size = _raw_auto(cl, 0).size * ntim
    cross_size = _raw_cross(cl, 0).size
    struct = bb_xml
    if cfg.with_flags:
        struct += f'<flags size="{flags_size}" axes="{_axes(cl, "flags")}"/>'
    if cfg.with_actual_times:
        struct += (
            f'<actualTimes size="{nant * ntim}" axes="{_axes(cl, "actualTimes")}"/>'
            f'<actualDurations size="{nant * ntim}" '
            f'axes="{_axes(cl, "actualDurations")}"/>'
        )
    if not cfg.is_auto_only:
        struct += f'<crossData size="{cross_size}" axes="{_axes(cl, "crossData")}"/>'
    struct += (
        f'<autoData size="{auto_size}" axes="{_axes(cl, "autoData")}" '
        'normalized="false"/>'
    )
    apc_attr = f' apc="{" ".join(cfg.apc)}"' if not cfg.is_auto_only else ""
    title = (
        "ALMA Radiometer Data"
        if cfg.processor_type == "RADIOMETER"
        else "ALMA Correlator Spectral Data"
    )
    data_oid = "uid://A002/X1/X" + os.path.basename(path).rsplit("_X", 1)[-1]
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sdmDataHeader xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'byteOrder="Little_Endian" schemaVersion="2" projectPath="{eb}/{scan}/{subscan}/">\n'
        f"<startTime>{int(bdf.mid_ns[0] - bdf.interval_ns[0] // 2)}</startTime>\n"
        f'<dataOID xlink:href="{data_oid}" xlink:title="{title}"/>\n'
        + (
            f"<numTime>{cfg.packed_num_time}</numTime>\n"
            if packed
            else "<dimensionality>1</dimensionality>\n"
        )
        + '<execBlock xlink:href="uid://A002/X1234/X5678"/>\n'
        f"<numAntenna>{nant}</numAntenna>\n"
        f"<correlationMode>{cfg.correlation_mode}</correlationMode>\n"
        f"<spectralResolution>{cfg.spectral_type}</spectralResolution>\n"
        f"<processorType>{cfg.processor_type}</processorType>\n"
        f"<dataStruct{apc_attr}>{struct}</dataStruct>\n"
        "</sdmDataHeader>\n"
    )

    out = bytearray()
    out += b"MIME-Version: 1.0\n"
    out += (
        f'Content-Type: multipart/mixed; boundary="{_B1}"; type="text/xml"\n'.encode()
    )
    out += b"Content-Description: Correlator\n"
    out += f"Content-Location: uid://A002/X1/X{bdf.main_row}/\n".encode()
    out += b"\n"
    out += f"--{_B1}\n".encode()
    out += b'Content-Type: text/xml; charset="utf-8"\n'
    out += b"Content-Location: sdmDataHeader.xml\n"
    out += b"\n"
    out += header.encode()

    if packed:
        subsets = [
            (
                int(
                    bdf.mid_ns[0] - bdf.interval_ns[0] // 2 + bdf.interval_ns.sum() // 2
                ),
                int(bdf.interval_ns.sum()),
                list(range(ntim)),
            )
        ]
    else:
        subsets = [
            (int(bdf.mid_ns[i]), int(bdf.interval_ns[i]), [i])
            for i in range(bdf.num_times)
        ]

    for sub_idx, (sub_mid, sub_interval, tim_idxs) in enumerate(subsets):
        pp = f"{eb}/{scan}/{subscan}/{sub_idx + 1}/"
        out += f"--{_B1}\n".encode()
        out += (
            f'Content-Type: multipart/related; boundary="{_B2}"; type="text/xml"; '
            'start="<DataSubset.xml>"\n'
        ).encode()
        out += b"Content-Description: Data and metadata subset\n"
        out += f"--{_B2}\n".encode()
        out += b'Content-Type: text/xml; charset="utf-8"\n'
        out += f"Content-Location: {pp}desc.xml\n".encode()
        out += b"\n"
        components = []
        if cfg.with_flags:
            components.append(
                (
                    "flags",
                    np.concatenate([_raw_flags(cl, bdf.codes[i]) for i in tim_idxs]),
                )
            )
        if cfg.with_actual_times:
            components.append(
                (
                    "actualTimes",
                    np.repeat(bdf.mid_ns[tim_idxs], nant).astype("<i8"),
                )
            )
            components.append(
                (
                    "actualDurations",
                    np.repeat(bdf.interval_ns[tim_idxs], nant).astype("<i8"),
                )
            )
        if not cfg.is_auto_only:
            components.append(
                (
                    "crossData",
                    _cross_to_stored(cl, _raw_cross(cl, bdf.codes[tim_idxs[0]])),
                )
            )
        components.append(
            (
                "autoData",
                np.concatenate([_raw_auto(cl, bdf.codes[i]) for i in tim_idxs]),
            )
        )
        sub_xml = (
            '<sdmDataSubsetHeader xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'projectPath="{pp}">\n'
            f"<schedulePeriodTime><time>{sub_mid}</time>"
            f"<interval>{sub_interval}</interval></schedulePeriodTime>\n"
            '<dataStruct ref="sdmDataHeader"/>\n'
        )
        for name, _arr in components:
            type_attr = f' type="{cfg.cross_type}"' if name == "crossData" else ""
            sub_xml += f'<{name} xlink:href="{pp}{name}.bin"{type_attr}/>\n'
        sub_xml += "</sdmDataSubsetHeader>\n"
        out += sub_xml.encode()
        for name, arr in components:
            out += f"--{_B2}\n".encode()
            out += b"Content-Type: binary/octet-stream\n"
            out += f"Content-Location: {pp}{name}.bin\n".encode()
            out += b"\n"
            out += np.ascontiguousarray(arr).tobytes()
            out += b"\n"
        out += f"--{_B2}--\n".encode()
    out += f"--{_B1}--\n".encode()

    with open(path, "wb") as bdf_file:
        bdf_file.write(bytes(out))


# --------------------------------------------------------------------------
# ASDM tables
# --------------------------------------------------------------------------


#: a leaf XML element whose text is a boolean literal, e.g. "<aborted> false </aborted>"
_XML_BOOLEAN_ELEMENT = re.compile(r"<(\w+)>\s*(true|false)\s*</\1>", re.IGNORECASE)


def set_row_from_xml(row, xml: str):
    """
    ``row.setFromXML(xml)``, with boolean attributes set to the value in ``xml``.

    The pyasdm XML row parsers read boolean attributes with ``bool(text)``, so
    any non-empty text, "false" included, gives True (e.g.
    ``Pointing.usePolynomials``, ``ExecBlock.aborted``,
    ``SpectralWindow.quantization``). After parsing, every element of ``xml``
    whose text is ``true``/``false`` (any case) and whose attribute parsed as a
    bool is set again through the row's ``set<Attribute>`` setter, so that the
    in-memory rows, and binary tables written from them, carry the intended
    values. (A table written as XML still reads back as True with pyasdm.)

    Returns
    -------
    row
        The same row, for chaining.
    """
    row.setFromXML(xml)
    for name, text in _XML_BOOLEAN_ELEMENT.findall(xml):
        accessor = name[0].upper() + name[1:]
        getter = getattr(row, "get" + accessor, None)
        setter = getattr(row, "set" + accessor, None)
        if getter is None or setter is None or not isinstance(getter(), bool):
            # not a boolean attribute of this row (e.g. a string that reads "true")
            continue
        setter(text.lower() == "true")
    return row


def _add_row(table, row_cls, xml: str):
    """Add a row from XML. Returns the row actually in the table (pyasdm may dedup)."""
    return table.add(set_row_from_xml(row_cls(table), xml))


def _tag_value(tag) -> int:
    return int(tag.getTagValue())


def _fmt(value: float) -> str:
    return repr(float(value))


def _dir_xml(direction) -> str:
    return f"2 1 2 {_fmt(direction[0])} {_fmt(direction[1])}"


def _uid_for_main_row(row_idx: int) -> str:
    return f"uid://A002/X1/X{row_idx + 0x100:x}"


def _corr_product(corr: str) -> tuple[str, str]:
    return ("X", "X") if corr == "I" else (corr[0], corr[1])


def _build_tables(spec: ASDMSpec):
    """Build the pyasdm.ASDM (in memory) and the truth (without paths)."""
    asdm = pyasdm.ASDM()
    nant = spec.num_antenna
    if nant < 3:
        raise ValueError("Use at least 3 antennas")
    if nant > len(STATION_OFFSETS_ITRF):
        raise ValueError(f"At most {len(STATION_OFFSETS_ITRF)} antennas supported")

    # ---- Antenna / Station
    antenna_names = [f"DA{41 + idx}" for idx in range(nant)]
    station_names = [f"A{100 + idx:03d}" for idx in range(nant)]
    positions = ARRAY_CENTER_ITRF[None, :] + STATION_OFFSETS_ITRF[:nant]
    for idx in range(nant):
        pos = positions[idx]
        station = _add_row(
            asdm.getStation(),
            pyasdm.StationRow,
            f"""<row><stationId> Station_{idx} </stationId><name> {station_names[idx]} </name>
<position> 1 3 {_fmt(pos[0])} {_fmt(pos[1])} {_fmt(pos[2])} </position><type>ANTENNA_PAD</type></row>""",
        )
        assert _tag_value(station.getStationId()) == idx
        antenna = _add_row(
            asdm.getAntenna(),
            pyasdm.AntennaRow,
            f"""<row><antennaId> Antenna_{idx} </antennaId><name> {antenna_names[idx]} </name>
<antennaMake>AEM_12</antennaMake><antennaType>GROUND_BASED</antennaType><dishDiameter> 12.0 </dishDiameter>
<position> 1 3 0.0 0.0 0.0 </position><offset> 1 3 0.0 0.0 0.0 </offset>
<time> {spec.t0_ns} </time><stationId> Station_{idx} </stationId></row>""",
        )
        assert _tag_value(antenna.getAntennaId()) == idx

    # ---- SPW / Polarization / DataDescription
    configs = spec.configs
    next_dd = 0
    dd_assignment: list[list[int]] = []
    for cfg in configs:
        if cfg.dd_ids is not None:
            if len(cfg.dd_ids) != len(cfg.spws):
                raise ValueError("dd_ids must have one entry per SPW")
            dd_assignment.append(list(cfg.dd_ids))
        else:
            dd_assignment.append(list(range(next_dd, next_dd + len(cfg.spws))))
            next_dd += len(cfg.spws)
    all_dds = sorted(dd for dds in dd_assignment for dd in dds)
    if all_dds != list(range(len(all_dds))):
        raise ValueError(f"DD ids must be unique and contiguous from 0: {all_dds}")

    spw_by_dd: dict[int, tuple[int, int, int, int]] = {}
    for cfg_idx, cfg in enumerate(configs):
        bdf_index = 0
        for bb_idx, bb in enumerate(cfg.basebands):
            for idx_in_bb, _spw_spec in enumerate(bb.spws):
                spw_by_dd[dd_assignment[cfg_idx][bdf_index]] = (
                    cfg_idx,
                    bdf_index,
                    bb_idx,
                    idx_in_bb,
                )
                bdf_index += 1

    pol_ids: dict[tuple[str, ...], int] = {}
    spws_truth: list[SpwTruth] = []
    for dd in range(len(all_dds)):
        cfg_idx, bdf_index, bb_idx, idx_in_bb = spw_by_dd[dd]
        cfg = configs[cfg_idx]
        bb = cfg.basebands[bb_idx]
        spw_spec = bb.spws[idx_in_bb]
        pol = POL_SETUPS[spw_spec.pol]
        products = pol.sd if (cfg.is_auto_only or not pol.cross) else pol.cross
        if products not in pol_ids:
            corr_products = " ".join(" ".join(_corr_product(c)) for c in products)
            pol_row = _add_row(
                asdm.getPolarization(),
                pyasdm.PolarizationRow,
                f"""<row><polarizationId> Polarization_{len(pol_ids)} </polarizationId>
<numCorr> {len(products)} </numCorr><corrType> 1 {len(products)} {" ".join(products)}</corrType>
<corrProduct> 2 {len(products)} 2 {corr_products}</corrProduct></row>""",
            )
            assert _tag_value(pol_row.getPolarizationId()) == len(pol_ids)
            pol_ids[products] = len(pol_ids)

        start = spw_spec.chan_freq_start
        if start is None:
            start = 8.0e10 + 2.0e9 * dd + 1.0e8 * cfg_idx
        step = spw_spec.chan_freq_step
        chan_freqs = start + step * np.arange(spw_spec.nchan)
        ref_freq = start + step * (spw_spec.nchan // 2)
        bandwidth = abs(step) * spw_spec.nchan
        if spw_spec.use_chan_freq_array:
            freq_xml = (
                f"<chanFreqArray> 1 {spw_spec.nchan} "
                + " ".join(_fmt(freq) for freq in chan_freqs)
                + " </chanFreqArray>"
            )
        else:
            freq_xml = (
                f"<chanFreqStart> {_fmt(start)} </chanFreqStart>"
                f"<chanFreqStep> {_fmt(step)} </chanFreqStep>"
            )
        short = _SPECTRAL_TYPE_SHORT.get(cfg.spectral_type, "FULL_RES")
        if (
            cfg.processor_type == "RADIOMETER"
            and cfg.spectral_type == "FULL_RESOLUTION"
        ):
            short = "WVR"
        spw_name = f"X0000000000#ALMA_RB_03#{bb.name}#SW-{idx_in_bb + 1:02d}#{short}"
        spw_row = _add_row(
            asdm.getSpectralWindow(),
            pyasdm.SpectralWindowRow,
            f"""<row><spectralWindowId> SpectralWindow_{dd} </spectralWindowId>
<basebandName>{bb.name}</basebandName><netSideband>{spw_spec.sideband}</netSideband>
<numChan> {spw_spec.nchan} </numChan><numBin> {spw_spec.num_bin} </numBin>
<refFreq> {_fmt(ref_freq)} </refFreq><measFreqRef> TOPO </measFreqRef>
<sidebandProcessingMode>NONE</sidebandProcessingMode><totBandwidth> {_fmt(bandwidth)} </totBandwidth>
<windowFunction>HANNING</windowFunction>{freq_xml}
<chanWidth> {_fmt(abs(step))} </chanWidth><effectiveBw> {_fmt(abs(step))} </effectiveBw>
<name> {spw_name} </name><resolution> {_fmt(abs(step))} </resolution></row>""",
        )
        assert _tag_value(spw_row.getSpectralWindowId()) == dd, "SPW deduplicated"
        dd_row = _add_row(
            asdm.getDataDescription(),
            pyasdm.DataDescriptionRow,
            f"""<row><dataDescriptionId> DataDescription_{dd} </dataDescriptionId>
<polOrHoloId> Polarization_{pol_ids[products]} </polOrHoloId>
<spectralWindowId> SpectralWindow_{dd} </spectralWindowId></row>""",
        )
        assert _tag_value(dd_row.getDataDescriptionId()) == dd
        spws_truth.append(
            SpwTruth(
                spw_id=dd,
                dd_id=dd,
                config_idx=cfg_idx,
                bdf_index=bdf_index,
                baseband_name=bb.name,
                baseband_index=bb_idx,
                index_in_baseband=idx_in_bb,
                nchan=spw_spec.nchan,
                pol=pol,
                num_bin=spw_spec.num_bin,
                chan_freqs=chan_freqs,
                correlation_mode=cfg.correlation_mode,
                polarization_products=products,
            )
        )

    # ---- Processor / ConfigDescription
    for cfg_idx, cfg in enumerate(configs):
        sub_type = (
            "ALMA_CORRELATOR_MODE"
            if cfg.processor_type != "RADIOMETER"
            else (
                "ALMA_RADIOMETER"
                if cfg.spectral_type == "FULL_RESOLUTION"
                else "SQUARE_LAW_DETECTOR"
            )
        )
        proc = _add_row(
            asdm.getProcessor(),
            pyasdm.ProcessorRow,
            f"""<row><processorId> Processor_{cfg_idx} </processorId><modeId> CorrelatorMode_{cfg_idx} </modeId>
<processorType>{cfg.processor_type}</processorType><processorSubType>{sub_type}</processorSubType></row>""",
        )
        assert _tag_value(proc.getProcessorId()) == cfg_idx
        dds = dd_assignment[cfg_idx]
        apc = cfg.apc if not cfg.is_auto_only else ("AP_UNCORRECTED",)
        ants = " ".join(f"Antenna_{idx}" for idx in range(nant))
        cfg_row = _add_row(
            asdm.getConfigDescription(),
            pyasdm.ConfigDescriptionRow,
            f"""<row><numAntenna> {nant} </numAntenna><numDataDescription> {len(dds)} </numDataDescription>
<numFeed> 1 </numFeed><correlationMode>{cfg.correlation_mode}</correlationMode>
<configDescriptionId> ConfigDescription_{cfg_idx} </configDescriptionId>
<numAtmPhaseCorrection> {len(apc)} </numAtmPhaseCorrection>
<atmPhaseCorrection> 1 {len(apc)} {" ".join(apc)}</atmPhaseCorrection>
<processorType>{cfg.processor_type}</processorType><spectralType>{cfg.spectral_type}</spectralType>
<antennaId> 1 {nant} {ants} </antennaId>
<dataDescriptionId> 1 {len(dds)} {" ".join(f"DataDescription_{dd}" for dd in dds)} </dataDescriptionId>
<feedId> 1 {nant} {" ".join(["0"] * nant)} </feedId><processorId> Processor_{cfg_idx} </processorId>
<switchCycleId> 1 {len(dds)} {" ".join(["SwitchCycle_0"] * len(dds))} </switchCycleId></row>""",
        )
        assert _tag_value(cfg_row.getConfigDescriptionId()) == cfg_idx

    # ---- Feed (one per antenna and SPW)
    for ant in range(nant):
        for spw in spws_truth:
            _add_row(
                asdm.getFeed(),
                pyasdm.FeedRow,
                f"""<row><feedId> 0 </feedId><timeInterval> 7226686294548387903 3993371484612775807 </timeInterval>
<numReceptor> 2 </numReceptor><beamOffset> 2 2 2 0.0 0.0 0.0 0.0 </beamOffset>
<focusReference> 2 2 3 0.0 0.0 0.0 0.0 0.0 0.0 </focusReference><polarizationTypes> 1 2 X Y</polarizationTypes>
<polResponse> 2 2 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 </polResponse>
<receptorAngle> 1 2 -0.9346238144 0.6361725124 </receptorAngle><antennaId> Antenna_{ant} </antennaId>
<receiverId> 1 2 0 0 </receiverId><spectralWindowId> SpectralWindow_{spw.spw_id} </spectralWindowId></row>""",
            )

    # ---- Field / Source
    source_names: list[str] = []
    field_source_ids: list[int] = []
    for fld in spec.fields:
        src = fld.source_name or fld.name
        if src not in source_names:
            source_names.append(src)
        field_source_ids.append(source_names.index(src))
    for field_id, fld in enumerate(spec.fields):
        ref_dir = fld.reference_dir or fld.phase_dir
        delay_dir = fld.delay_dir or fld.phase_dir
        field_row = _add_row(
            asdm.getField(),
            pyasdm.FieldRow,
            f"""<row><fieldId> Field_{field_id} </fieldId><fieldName> {fld.name} </fieldName><numPoly> 1 </numPoly>
<delayDir> {_dir_xml(delay_dir)} </delayDir><phaseDir> {_dir_xml(fld.phase_dir)} </phaseDir>
<referenceDir> {_dir_xml(ref_dir)} </referenceDir><time> {spec.t0_ns} </time><code> none </code>
<directionCode>{fld.direction_code}</directionCode><sourceId> {field_source_ids[field_id]} </sourceId></row>""",
        )
        assert _tag_value(field_row.getFieldId()) == field_id, "Field deduplicated"
    for src_id, src in enumerate(source_names):
        fld = spec.fields[field_source_ids.index(src_id)]
        for spw in spws_truth:
            src_row = _add_row(
                asdm.getSource(),
                pyasdm.SourceRow,
                f"""<row><sourceId> {src_id} </sourceId>
<timeInterval> 7090683272335387903 4265377529038775807 </timeInterval><code> none </code>
<direction> 1 2 {_fmt(fld.phase_dir[0])} {_fmt(fld.phase_dir[1])} </direction>
<properMotion> 1 2 0.0 0.0 </properMotion><sourceName> {src} </sourceName>
<directionCode>ICRS</directionCode><spectralWindowId> SpectralWindow_{spw.spw_id} </spectralWindowId></row>""",
            )
            assert src_row.getSourceId() == src_id

    # ---- State
    state_ids: dict[str, int] = {}
    for scan in spec.scans:
        for sub in scan.subscans:
            if sub.intent in state_ids:
                continue
            device, sig, ref, on_sky = _SUBSCAN_INTENT_STATE[sub.intent]
            state = _add_row(
                asdm.getState(),
                pyasdm.StateRow,
                f"""<row><stateId> State_{len(state_ids)} </stateId><calDeviceName>{device}</calDeviceName>
<sig> {sig} </sig><ref> {ref} </ref><onSky> {on_sky} </onSky></row>""",
            )
            state_ids[sub.intent] = _tag_value(state.getStateId())

    # ---- timeline: Scan / Subscan / Main / BDF truth
    bdfs: list[BDFTruth] = []
    codes_next = [0] * len(configs)
    tnow = spec.t0_ns
    main_row_idx = 0
    pointing_rows = []  # (antenna, start, duration)
    for scan_idx, scan in enumerate(spec.scans):
        scan_number = scan_idx + 1
        scan_start = tnow
        for sub_idx, sub in enumerate(scan.subscans):
            subscan_number = sub_idx + 1
            duration = sub.num_integrations * spec.integration_ns
            sub_start = tnow
            sub_mid = sub_start + duration // 2
            for cfg_idx, cfg in enumerate(configs):
                if cfg.packed_num_time:
                    ntimes = cfg.packed_num_time
                else:
                    ntimes = cfg.num_integrations or sub.num_integrations
                if duration % ntimes or (duration // ntimes) % 2:
                    raise ValueError("integration_ns does not split evenly")
                dt = duration // ntimes
                mid_ns = sub_start + dt // 2 + dt * np.arange(ntimes, dtype=np.int64)
                codes = np.arange(codes_next[cfg_idx], codes_next[cfg_idx] + ntimes)
                codes_next[cfg_idx] += ntimes
                bdfs.append(
                    BDFTruth(
                        path="",
                        main_row=main_row_idx,
                        config_idx=cfg_idx,
                        scan_number=scan_number,
                        subscan_number=subscan_number,
                        field_id=sub.field_id,
                        scan_intents=tuple(scan.intents),
                        subscan_intent=sub.intent,
                        main_time_ns=int(sub_mid),
                        mid_ns=mid_ns,
                        interval_ns=np.full(ntimes, dt, dtype=np.int64),
                        codes=codes,
                        packed=bool(cfg.packed_num_time),
                    )
                )
                states = " ".join([f"State_{state_ids[sub.intent]}"] * nant)
                _add_row(
                    asdm.getMain(),
                    pyasdm.MainRow,
                    f"""<row><time> {sub_mid} </time><numAntenna> {nant} </numAntenna>
<timeSampling>INTEGRATION</timeSampling><interval> {duration} </interval>
<numIntegration> {ntimes} </numIntegration><scanNumber> {scan_number} </scanNumber>
<subscanNumber> {subscan_number} </subscanNumber><dataSize> 1000 </dataSize>
<dataUID><EntityRef entityId="{_uid_for_main_row(main_row_idx)}" partId="X00000000" entityTypeName="Main" documentVersion="1"/></dataUID>
<configDescriptionId> ConfigDescription_{cfg_idx} </configDescriptionId><execBlockId> ExecBlock_0 </execBlockId>
<fieldId> Field_{sub.field_id} </fieldId><stateId> 1 {nant} {states} </stateId></row>""",
                )
                main_row_idx += 1
            _add_row(
                asdm.getSubscan(),
                pyasdm.SubscanRow,
                f"""<row><scanNumber> {scan_number} </scanNumber><subscanNumber> {subscan_number} </subscanNumber>
<startTime> {sub_start} </startTime><endTime> {sub_start + duration} </endTime>
<fieldName> {spec.fields[sub.field_id].name} </fieldName><subscanIntent>{sub.intent}</subscanIntent>
<subscanMode>NO_SWITCHING</subscanMode><numIntegration> {sub.num_integrations} </numIntegration>
<numSubintegration> 1 {sub.num_integrations} {" ".join(["0"] * sub.num_integrations)} </numSubintegration>
<correlatorCalibration>NONE</correlatorCalibration><execBlockId> ExecBlock_0 </execBlockId></row>""",
            )
            pointing_rows.append((sub_start, duration))
            tnow = sub_start + duration + spec.subscan_gap_ns
        scan_end = tnow - spec.subscan_gap_ns
        field_names_in_scan = sorted(
            {spec.fields[sub.field_id].name for sub in scan.subscans}
        )
        quoted_names = " ".join(f"&quot;{name}&quot;" for name in field_names_in_scan)
        nint = len(scan.intents)
        _add_row(
            asdm.getScan(),
            pyasdm.ScanRow,
            f"""<row><scanNumber> {scan_number} </scanNumber><startTime> {scan_start} </startTime>
<endTime> {scan_end} </endTime><numIntent> {nint} </numIntent><numSubscan> {len(scan.subscans)} </numSubscan>
<scanIntent> 1 {nint} {" ".join(scan.intents)}</scanIntent><calDataType> 1 {nint} {" ".join(["NONE"] * nint)}</calDataType>
<calibrationOnLine> 1 {nint} {" ".join(["false"] * nint)} </calibrationOnLine>
<numField> {len(field_names_in_scan)} </numField><fieldName> 1 {len(field_names_in_scan)} {quoted_names} </fieldName>
<sourceName> {spec.fields[scan.subscans[0].field_id].source_name or spec.fields[scan.subscans[0].field_id].name} </sourceName>
<execBlockId> ExecBlock_0 </execBlockId></row>""",
        )
        tnow = scan_end + spec.scan_gap_ns
    end_time = tnow

    # ---- ExecBlock / SBSummary
    ants = " ".join(f"Antenna_{idx}" for idx in range(nant))
    release_xml = (
        f"<releaseDate> {spec.release_date_ns} </releaseDate>"
        if spec.release_date_ns is not None
        else ""
    )
    _add_row(
        asdm.getExecBlock(),
        pyasdm.ExecBlockRow,
        f"""<row><execBlockId> ExecBlock_0 </execBlockId><startTime> {spec.t0_ns - 10**9} </startTime>
<endTime> {end_time} </endTime><execBlockNum> {spec.exec_block_num} </execBlockNum>
<execBlockUID><EntityRef entityId="uid://A002/X1234/X5678" partId="X00000000" entityTypeName="ASDM" documentVersion="1"/></execBlockUID>
<projectUID><EntityRef entityId="uid://A001/X35fd/X21f" partId="X00000000" entityTypeName="ObsProject" documentVersion="1"/></projectUID>
<configName> C-1 </configName><telescopeName> ALMA </telescopeName><observerName> synthetic </observerName>
<numObservingLog> 0 </numObservingLog><observingLog> 1 0 </observingLog>
<sessionReference><EntityRef entityId="uid://A002/X1234/X5677" partId="X00000000" entityTypeName="Session" documentVersion="1"/></sessionReference>
<baseRangeMin> 0.0 </baseRangeMin><baseRangeMax> 0.0 </baseRangeMax><baseRmsMinor> 0.0 </baseRmsMinor>
<baseRmsMajor> 0.0 </baseRmsMajor><basePa> 0.0 </basePa><aborted> false </aborted>
<numAntenna> {nant} </numAntenna>{release_xml}<observingScript> synthetic.py </observingScript>
<antennaId> 1 {nant} {ants} </antennaId><sBSummaryId> SBSummary_0 </sBSummaryId></row>""",
    )
    _add_row(
        asdm.getSBSummary(),
        pyasdm.SBSummaryRow,
        """<row><sBSummaryId> SBSummary_0 </sBSummaryId>
<sbSummaryUID><EntityRef entityId="uid://A001/X362e/X332" partId="X00000000" entityTypeName="SchedBlock" documentVersion="1"/></sbSummaryUID>
<projectUID><EntityRef entityId="uid://A001/X35fd/X21f" partId="X00000000" entityTypeName="ObsProject" documentVersion="1"/></projectUID>
<obsUnitSetUID><EntityRef entityId="uid://A001/X35fd/X21f" partId="X00000000" entityTypeName="ObsProject" documentVersion="1"/></obsUnitSetUID>
<frequency> 100.0 </frequency><frequencyBand>ALMA_RB_03</frequencyBand><sbType>OBSERVER</sbType>
<sbDuration> 3600000000000 </sbDuration><numObservingMode> 1 </numObservingMode>
<observingMode> 1 1 &quot;Standard Interferometry&quot; </observingMode><numberRepeats> 1 </numberRepeats>
<numScienceGoal> 1 </numScienceGoal><scienceGoal> 1 1 &quot;representativeFrequency = 100.0 GHz&quot; </scienceGoal>
<numWeatherConstraint> 1 </numWeatherConstraint><weatherConstraint> 1 1 &quot;maxPWVC = 1.0 mm&quot; </weatherConstraint>
</row>""",
    )

    truth = SyntheticASDM(
        path="",
        spec=spec,
        spws=spws_truth,
        bdfs=bdfs,
        antenna_names=antenna_names,
        station_names=station_names,
        antenna_positions=positions,
        field_names=[fld.name for fld in spec.fields],
        source_names=source_names,
        field_source_ids=field_source_ids,
    )

    # ---- Pointing (optional): one row per antenna and subscan
    if spec.with_pointing:
        _add_pointing(asdm, spec, truth, pointing_rows)

    return asdm, truth


def _pointing_array_xml(values: np.ndarray) -> str:
    return f"2 {values.shape[0]} 2 " + " ".join(_fmt(val) for val in values.ravel())


def _add_pointing(asdm, spec: ASDMSpec, truth: SyntheticASDM, rows) -> None:
    nsample = spec.pointing_num_sample
    times, targets, encoders = [], [], []
    global_sample = 0
    for start, duration in rows:
        dt = duration // nsample
        sample_mid = start + dt // 2 + dt * np.arange(nsample, dtype=np.int64)
        sample_idx = global_sample + np.arange(nsample)
        global_sample += nsample
        times.append((sample_mid - ASDM_TO_UNIX_OFFSET_NS) / 1e9)
        row_targets, row_encoders = [], []
        for ant in range(spec.num_antenna):
            az = 1.0 + 1.0e-3 * sample_idx + 0.01 * ant
            alt = 0.9 - 1.0e-4 * sample_idx
            target = np.stack([az, alt], axis=-1)
            encoder = target + 1.0e-5
            if spec.pointing_antennas is not None and ant not in spec.pointing_antennas:
                row_targets.append(np.full_like(target, np.nan))
                row_encoders.append(np.full_like(encoder, np.nan))
                continue
            row_targets.append(target)
            row_encoders.append(encoder)
            zeros = np.zeros((nsample, 2))
            # sampled values, usePolynomials=False (_add_row sets the booleans
            # as written, pyasdm's XML parser alone would read "false" as True)
            _add_row(
                asdm.getPointing(),
                pyasdm.PointingRow,
                f"""<row><antennaId> Antenna_{ant} </antennaId>
<timeInterval> {start + duration // 2} {duration} </timeInterval><numSample> {nsample} </numSample>
<encoder> {_pointing_array_xml(encoder)} </encoder><pointingTracking> true </pointingTracking>
<usePolynomials> false </usePolynomials><timeOrigin> {start} </timeOrigin><numTerm> {nsample} </numTerm>
<pointingDirection> {_pointing_array_xml(encoder)} </pointingDirection><target> {_pointing_array_xml(target)} </target>
<offset> {_pointing_array_xml(zeros)} </offset><sourceOffset> {_pointing_array_xml(zeros)} </sourceOffset>
<pointingModelId> 0 </pointingModelId></row>""",
            )
        targets.append(np.stack(row_targets, axis=1))
        encoders.append(np.stack(row_encoders, axis=1))
    truth.pointing_times_unix = np.concatenate(times)
    truth.pointing_target = np.concatenate(targets)
    truth.pointing_encoder = np.concatenate(encoders)
    if not spec.pointing_as_bin:
        asdm.getPointing()._fileAsBin = False


@contextlib.contextmanager
def _pyasdm_complex_str_workaround():
    """
    pyasdm 0.0.7 ``Complex.__str__`` references an undefined name (``asdm``), so
    writing a table with Complex attributes (Feed.polResponse) to XML fails.
    Patch it only while writing, and only if the bug is present.
    """
    complex_cls = pyasdm.types.Complex
    try:
        str(complex_cls(0.0, 0.0))
        buggy = False
    except NameError:
        buggy = True
    if not buggy:
        yield
        return
    original = complex_cls.__str__

    def _str(self):
        return (
            pyasdm.Parser.doubleToString(self.getReal())
            + " "
            + pyasdm.Parser.doubleToString(self.getImg())
        )

    complex_cls.__str__ = _str
    try:
        yield
    finally:
        complex_cls.__str__ = original


def write_synthetic_asdm(spec: ASDMSpec, directory: str) -> SyntheticASDM:
    """
    Write a synthetic ASDM to disk.

    Parameters
    ----------
    spec : ASDMSpec
        Description of the ASDM (see the ``*_spec()`` factories).
    directory : str
        Parent directory. The ASDM is written as ``<directory>/<spec.name>``:
        tables with pyasdm ``ASDM.toFile`` and one MIME BDF per Main row in
        ``ASDMBinary/`` (at the path ``MainRow.getBDFPath()`` resolves to).

    Returns
    -------
    SyntheticASDM
        The truth: what was written, with the expected-value helpers.
    """
    asdm, truth = _build_tables(spec)
    asdm_dir = os.path.join(os.path.abspath(directory), spec.name)
    with contextlib.redirect_stdout(io.StringIO()), _pyasdm_complex_str_workaround():
        asdm.toFile(asdm_dir)
    bin_dir = os.path.join(asdm_dir, "ASDMBinary")
    os.makedirs(bin_dir, exist_ok=True)
    truth.path = asdm_dir

    layouts = []
    for cfg_idx, cfg in enumerate(spec.configs):
        spws = sorted(
            [spw for spw in truth.spws if spw.config_idx == cfg_idx],
            key=lambda spw: spw.bdf_index,
        )
        layouts.append(
            _ConfigLayout(
                cfg=cfg,
                spws=spws,
                layout=code_layout(spec, cfg_idx),
                num_antenna=spec.num_antenna,
            )
        )
    for bdf in truth.bdfs:
        name = _uid_for_main_row(bdf.main_row).replace("/", "_").replace(":", "_")
        bdf.path = os.path.join(bin_dir, name)
        _write_bdf(bdf.path, layouts[bdf.config_idx], bdf, spec)
    return truth


# --------------------------------------------------------------------------
# Read-back verification (with pyasdm only)
# --------------------------------------------------------------------------


def verify_bdf_roundtrip(truth: SyntheticASDM) -> None:
    """
    Check a written synthetic ASDM with pyasdm only.

    The tables are read back with ``ASDM.setFromFile`` and every BDF with
    ``pyasdm.bdf.BDFReader.getSubset``; headers and raw arrays must be those
    written.

    Parameters
    ----------
    truth : SyntheticASDM
        Truth returned by ``write_synthetic_asdm``.

    Raises
    ------
    AssertionError
        On any mismatch.
    """
    asdm = pyasdm.ASDM()
    with contextlib.redirect_stdout(io.StringIO()):
        asdm.setFromFile(truth.path)
    main_rows = asdm.getMain().get()
    assert len(main_rows) == len(truth.bdfs)
    for row, bdf in zip(main_rows, truth.bdfs, strict=True):
        assert os.path.abspath(row.getBDFPath()) == bdf.path, (
            row.getBDFPath(),
            bdf.path,
        )
        assert row.getTime().get() == bdf.main_time_ns
    spw_rows = asdm.getSpectralWindow().get()
    assert [_tag_value(row.getSpectralWindowId()) for row in spw_rows] == [
        spw.spw_id for spw in truth.spws
    ]

    for bdf in truth.bdfs:
        cfg = truth.config(bdf.config_idx)
        spws = sorted(
            [spw for spw in truth.spws if spw.config_idx == bdf.config_idx],
            key=lambda spw: spw.bdf_index,
        )
        cl = _ConfigLayout(
            cfg=cfg,
            spws=spws,
            layout=code_layout(truth.spec, bdf.config_idx),
            num_antenna=truth.num_antenna,
        )
        reader = pyasdm.bdf.BDFReader()
        reader.open(bdf.path)
        try:
            header = reader.getHeader()
            assert header.getNumAntenna() == truth.num_antenna
            assert header.getCorrelationMode().getName() == cfg.correlation_mode
            assert header.getDimensionality() == (0 if bdf.packed else 1)
            assert header.getNumTime() == (cfg.packed_num_time if bdf.packed else 0)
            basebands = header.getBasebandsList()
            assert [bb["name"] for bb in basebands] == [bb.name for bb in cfg.basebands]
            hdr_spws = [spw for bb in basebands for spw in bb["spectralWindows"]]
            assert [spw["numSpectralPoint"] for spw in hdr_spws] == [
                spw.nchan for spw in spws
            ]
            subsets = []
            while reader.hasSubset():
                subsets.append(reader.getSubset())
        finally:
            reader.close()
        if bdf.packed:
            assert len(subsets) == 1
            tim_groups = [list(range(cfg.packed_num_time))]
        else:
            assert len(subsets) == bdf.num_times
            tim_groups = [[idx] for idx in range(bdf.num_times)]
        for subset, tim_idxs in zip(subsets, tim_groups, strict=True):
            if not bdf.packed:
                assert subset["midpointInNanoSeconds"] == bdf.mid_ns[tim_idxs[0]]
                assert subset["intervalInNanoSeconds"] == bdf.interval_ns[tim_idxs[0]]
            auto = np.concatenate([_raw_auto(cl, bdf.codes[idx]) for idx in tim_idxs])
            np.testing.assert_array_equal(subset["autoData"]["arr"], auto)
            if cfg.with_flags:
                flags = np.concatenate(
                    [_raw_flags(cl, bdf.codes[idx]) for idx in tim_idxs]
                )
                np.testing.assert_array_equal(subset["flags"]["arr"], flags)
            if not cfg.is_auto_only:
                cross = _cross_to_stored(cl, _raw_cross(cl, bdf.codes[tim_idxs[0]]))
                assert subset["crossData"]["type"] == cfg.cross_type
                np.testing.assert_array_equal(subset["crossData"]["arr"], cross)
            if cfg.with_actual_times:
                np.testing.assert_array_equal(
                    subset["actualTimes"]["arr"],
                    np.repeat(bdf.mid_ns[tim_idxs], truth.num_antenna),
                )


def decode_raw_cross_subset(
    truth: SyntheticASDM, bdf: BDFTruth, raw: np.ndarray, spw_id: int
) -> np.ndarray:
    """
    Independent decoder of a raw crossData subset array (as returned by pyasdm
    getSubset) into (baseline, chan, pol) complex values of one SPW (APC
    AP_UNCORRECTED or the only APC, BIN 0). Used to cross-check the expected-value
    functions against the bytes on disk.
    """
    cfg = truth.config(bdf.config_idx)
    spws = sorted(
        [spw for spw in truth.spws if spw.config_idx == bdf.config_idx],
        key=lambda spw: spw.bdf_index,
    )
    napc = len(cfg.apc)
    apc_idx = cfg.loaded_apc_index
    block_sizes = [
        spw.num_bin * napc * spw.nchan * len(spw.pol.cross) * 2 for spw in spws
    ]
    per_bl = sum(block_sizes)
    target = truth.spw(spw_id)
    start = sum(block_sizes[: target.bdf_index])
    npol = len(target.pol.cross)
    raw = np.asarray(raw, dtype=float)
    nbl = len(truth.cross_baselines)
    out = np.zeros((nbl, target.nchan, npol), dtype=complex)
    for bl in range(nbl):
        block = raw[
            bl * per_bl + start : bl * per_bl + start + block_sizes[target.bdf_index]
        ]
        block = block.reshape(target.num_bin, napc, target.nchan, npol, 2)[0, apc_idx]
        out[bl] = block[..., 0] + 1j * block[..., 1]
    if CROSS_NP_TYPES[cfg.cross_type].kind == "i":
        out = out / cfg.scale_factor
    return out


# --------------------------------------------------------------------------
# Canonical layouts
# --------------------------------------------------------------------------

#: default target / calibrator directions (rad, ICRS)
_TARGET_DIR = (1.1487030439690096, -0.023431362760917465)
_CAL_DIR = (1.3528024488371877, 0.31436086058385826)


def interferometric_spec(
    cross_type: str = "FLOAT32_TYPE",
    scale_factor: float = 1.0,
    num_antenna: int = 4,
    with_pointing: bool = False,
    name: str = "uid___A002_X1234_X5678",
) -> ASDMSpec:
    """
    Interferometric, CROSS_AND_AUTO, dual-pol (XX, YY), 2 basebands with
    uneven SPW counts: BB_1 = [8 channels, 1 channel], BB_2 = [4 channels].
    One field; 2 scans x 2 subscans (all OBSERVE_TARGET#ON_SOURCE) with 3, 2,
    2, 3 integrations, i.e. one partition per SPW with 10 integrations in 4 BDFs.
    """
    cfg = ConfigSpec(
        basebands=[
            BasebandSpec("BB_1", [SpwSpec(8, "dual"), SpwSpec(1, "dual")]),
            BasebandSpec(
                "BB_2",
                [SpwSpec(4, "dual", chan_freq_step=-31.25e6, sideband="LSB")],
            ),
        ],
        cross_type=cross_type,
        scale_factor=scale_factor,
    )
    return ASDMSpec(
        configs=[cfg],
        scans=[
            ScanSpec([SubscanSpec(0, "ON_SOURCE", 3), SubscanSpec(0, "ON_SOURCE", 2)]),
            ScanSpec([SubscanSpec(0, "ON_SOURCE", 2), SubscanSpec(0, "ON_SOURCE", 3)]),
        ],
        fields=[FieldSpec("J0423-0120", _TARGET_DIR)],
        num_antenna=num_antenna,
        with_pointing=with_pointing,
        name=name,
    )


def small_dtype_spec(
    cross_type: str,
    scale_factor: float,
    apc: tuple[str, ...] = ("AP_UNCORRECTED",),
    num_bin: int = 1,
    name: str = "uid___A002_X1234_X7777",
) -> ASDMSpec:
    """Tiny interferometric ASDM (3 antennas, 1 dual-pol SPW of 4 channels, 3 integrations)."""
    cfg = ConfigSpec(
        basebands=[BasebandSpec("BB_1", [SpwSpec(4, "dual", num_bin=num_bin)])],
        cross_type=cross_type,
        scale_factor=scale_factor,
        apc=apc,
    )
    return ASDMSpec(
        configs=[cfg],
        scans=[ScanSpec([SubscanSpec(0, "ON_SOURCE", 3)])],
        fields=[FieldSpec("J0423-0120", _TARGET_DIR)],
        num_antenna=3,
        name=name,
    )


def full_pol_spec(name: str = "uid___A002_X1234_X5679") -> ASDMSpec:
    """
    Interferometric full-pol (cross XX XY YX YY, sd XX XY YY), INT32 data with
    scale factor 4, 2 basebands x 1 SPW (4 and 2 channels), 3 antennas, with
    actualTimes/actualDurations and an ExecBlock releaseDate. 1 scan with 2
    subscans (2 + 3 integrations) and two scan intents. The ConfigDescription
    lists its DDs out of DataDescription-table order ([1, 0], as after a retune
    reusing earlier DD rows), so the BDF SPW position differs from the DD order.
    """
    cfg = ConfigSpec(
        basebands=[
            BasebandSpec("BB_1", [SpwSpec(4, "full")]),
            BasebandSpec("BB_2", [SpwSpec(2, "full")]),
        ],
        cross_type="INT32_TYPE",
        scale_factor=4.0,
        with_actual_times=True,
        dd_ids=[1, 0],
    )
    release_date_ns = DEFAULT_T0_NS + 365 * 86400 * 10**9
    return ASDMSpec(
        configs=[cfg],
        scans=[
            ScanSpec(
                [SubscanSpec(0, "ON_SOURCE", 2), SubscanSpec(0, "ON_SOURCE", 3)],
                intents=("CALIBRATE_POLARIZATION", "CALIBRATE_WVR"),
            )
        ],
        fields=[FieldSpec("J1924-2914", _CAL_DIR)],
        num_antenna=3,
        name=name,
        release_date_ns=release_date_ns,
    )


def single_dish_spec(
    name: str = "uid___A002_X1234_X5680", with_off_and_calibration: bool = True
) -> ASDMSpec:
    """
    Single dish: every config AUTO_ONLY (CORRELATOR, FULL_RESOLUTION), 3
    antennas, 2 basebands x 1 dual-pol SPW (4 and 2 channels). Scan 1
    OBSERVE_TARGET with ON/OFF/ON subscans, scan 2 CALIBRATE_ATMOSPHERE with
    HOT/AMBIENT subscans. The field has referenceDir != phaseDir.
    With ``with_off_and_calibration=False``: a single scan with two ON_SOURCE
    subscans (one partition per SPW).
    """
    cfg = ConfigSpec(
        basebands=[
            BasebandSpec("BB_1", [SpwSpec(4, "dual")]),
            BasebandSpec("BB_2", [SpwSpec(2, "dual")]),
        ],
        correlation_mode="AUTO_ONLY",
    )
    if with_off_and_calibration:
        scans = [
            ScanSpec(
                [
                    SubscanSpec(0, "ON_SOURCE", 2),
                    SubscanSpec(0, "OFF_SOURCE", 2),
                    SubscanSpec(0, "ON_SOURCE", 3),
                ],
                intents=("OBSERVE_TARGET",),
            ),
            ScanSpec(
                [SubscanSpec(0, "HOT", 2), SubscanSpec(0, "AMBIENT", 2)],
                intents=("CALIBRATE_ATMOSPHERE",),
            ),
        ]
    else:
        scans = [
            ScanSpec(
                [SubscanSpec(0, "ON_SOURCE", 2), SubscanSpec(0, "ON_SOURCE", 3)],
                intents=("OBSERVE_TARGET",),
            )
        ]
    return ASDMSpec(
        configs=[cfg],
        scans=scans,
        fields=[
            FieldSpec(
                "Orion",
                _TARGET_DIR,
                reference_dir=(_TARGET_DIR[0] + 0.01, _TARGET_DIR[1] - 0.02),
            )
        ],
        num_antenna=3,
        name=name,
    )


def mosaic_spec(name: str = "uid___A002_X1234_X5681") -> ASDMSpec:
    """
    Interferometric dual-pol (1 SPW, 4 channels), 3 antennas. Fields: 0 = a
    calibrator, 1 and 2 = two mosaic pointings that share the fieldName
    "M100". Field 1 has phaseDir != referenceDir. Scans: 1 CALIBRATE_PHASE on
    field 0; 2 OBSERVE_TARGET with one subscan on field 1 and one on field 2;
    3 CALIBRATE_PHASE on field 0.
    """
    cfg = ConfigSpec(basebands=[BasebandSpec("BB_1", [SpwSpec(4, "dual")])])
    return ASDMSpec(
        configs=[cfg],
        scans=[
            ScanSpec([SubscanSpec(0, "ON_SOURCE", 2)], intents=("CALIBRATE_PHASE",)),
            ScanSpec(
                [SubscanSpec(1, "ON_SOURCE", 3), SubscanSpec(2, "ON_SOURCE", 2)],
                intents=("OBSERVE_TARGET",),
            ),
            ScanSpec([SubscanSpec(0, "ON_SOURCE", 2)], intents=("CALIBRATE_PHASE",)),
        ],
        fields=[
            FieldSpec("J1229+0203", _CAL_DIR),
            FieldSpec(
                "M100",
                (3.2387, 0.2738),
                reference_dir=(3.2387 + 0.05, 0.2738 - 0.03),
            ),
            FieldSpec("M100", (3.2391, 0.2741)),
        ],
        num_antenna=3,
        name=name,
    )


def interleaved_configs_spec(
    num_subscans: int = 6, name: str = "uid___A002_X1234_X5682"
) -> ASDMSpec:
    """
    ALMA-like interleaved ConfigDescriptions (the layout that triggered F07):
    0 = WVR (RADIOMETER, FULL_RESOLUTION, AUTO_ONLY, packed numTime=4, 1 DD),
    1 = SQLD (RADIOMETER, BASEBAND_WIDE, AUTO_ONLY, 4 DDs, 1 channel),
    2 = CHANNEL_AVERAGE (CORRELATOR, 4 DDs, 1 channel),
    3 = FULL_RESOLUTION (CORRELATOR, 4 DDs).
    The DD table interleaves CH_AVG and FULL_RES DDs (FULL_RES = 5, 7, 9, 11).
    One scan; every subscan has one Main row / BDF per config.
    """
    bbs = ["BB_1", "BB_2", "BB_3", "BB_4"]
    wvr = ConfigSpec(
        basebands=[
            BasebandSpec(
                "NOBB",
                [SpwSpec(4, "stokes_i", chan_freq_start=1.83e11, sideband="DSB")],
            )
        ],
        correlation_mode="AUTO_ONLY",
        processor_type="RADIOMETER",
        spectral_type="FULL_RESOLUTION",
        packed_num_time=4,
        dd_ids=[0],
    )
    sqld = ConfigSpec(
        basebands=[
            BasebandSpec(bb, [SpwSpec(1, "dual", chan_freq_step=2.0e9)]) for bb in bbs
        ],
        correlation_mode="AUTO_ONLY",
        processor_type="RADIOMETER",
        spectral_type="BASEBAND_WIDE",
        dd_ids=[1, 2, 3, 4],
    )
    ch_avg = ConfigSpec(
        basebands=[
            BasebandSpec(bb, [SpwSpec(1, "dual", chan_freq_step=1.875e9)]) for bb in bbs
        ],
        spectral_type="CHANNEL_AVERAGE",
        dd_ids=[6, 8, 10, 12],
    )
    full_res = ConfigSpec(
        basebands=[
            BasebandSpec(bb, [SpwSpec(nchan, "dual")])
            for bb, nchan in zip(bbs, [4, 2, 3, 2], strict=True)
        ],
        dd_ids=[5, 7, 9, 11],
    )
    return ASDMSpec(
        configs=[wvr, sqld, ch_avg, full_res],
        scans=[
            ScanSpec(
                [SubscanSpec(0, "ON_SOURCE", 2) for _ in range(num_subscans)],
                intents=("CALIBRATE_FOCUS", "CALIBRATE_WVR"),
            )
        ],
        fields=[FieldSpec("J0423-0120", _TARGET_DIR)],
        num_antenna=3,
        name=name,
    )
