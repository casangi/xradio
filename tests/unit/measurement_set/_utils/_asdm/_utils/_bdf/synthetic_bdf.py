"""
Minimal writer of synthetic BDFs (MIME multipart files parseable by
pyasdm.bdf.BDFReader), for value tests of the BDF loaders. Pytest-free helper,
loaded by file path from the test modules.

Layout written (BDF specification, outermost axis first)::

    crossData: [TIM] BAL BAB SPW [BIN] [APC] SPP POL (re, im)
    autoData:  [TIM] ANT BAB SPW [BIN] SPP POL  (sd XX XY YY: XX, Re XY, Im XY, YY)
    flags:     [TIM] BAL ANT BAB SPW POL  (BAL block followed by ANT block)

Every value encodes its position, so that the expected arrays are computed from
the definition of the BDF only (``expected_visibility``, ``expected_flags``),
independently of the xradio code:

- cross: true value ``re + 1j * im`` with ``re = idx + 1`` and
  ``im = -(2 * idx + 1)``, ``idx`` being the flat index of (integration,
  baseline, spw, bin, apc, channel, polarization). Integer data store the codes
  (true value = code / scaleFactor); float data store the true values.
- auto: float ``idx + 0.5`` (``idx`` flat index of (integration, antenna, spw,
  bin, channel, float within the channel)).
- flags: int32 words from ``FLAG_WORDS`` (several distinct bits, including the
  sign bit), flagged == (word != 0).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pyasdm

FLAG_WORDS = np.array(
    [0, 1, 0, 16, 0, 0, 2**30, 0, -(2**31), 0, 1 | 16 | 2**30, 0, 0], dtype=np.int32
)
CROSS_TYPES = {"FLOAT32_TYPE": "f4", "INT32_TYPE": "i4", "INT16_TYPE": "i2"}
MAX_CHAN = 16
_B1 = "MIME_boundary-1"
_B2 = "MIME_boundary-2"


@dataclass
class SpwDef:
    """One SPW of a baseband."""

    nchan: int
    cross_pols: tuple[str, ...] = ("XX", "YY")
    sd_pols: tuple[str, ...] = ("XX", "YY")
    num_bin: int = 1


@dataclass
class BDFDef:
    """Definition of a synthetic BDF."""

    basebands: list[list[SpwDef]]
    num_antenna: int = 3
    correlation_mode: str = "CROSS_AND_AUTO"
    processor_type: str = "CORRELATOR"
    cross_type: str = "FLOAT32_TYPE"
    scale_factor: float = 4.0
    apc: tuple[str, ...] = ("AP_UNCORRECTED",)
    #: number of subsets (non-packed BDFs: one integration per subset)
    num_subsets: int = 3
    #: > 0: packed BDF (dimensionality 0), one subset with this many TIM samples
    packed_num_time: int = 0
    with_flags: bool = True
    #: flags given per baseband (axes without SPW, pols of the first SPW)
    flags_per_baseband: bool = False
    big_endian: bool = False
    #: optional per-subset list of components to leave out ("flags", "crossData",
    #: "autoData")
    missing_components: dict[int, set[str]] = field(default_factory=dict)

    @property
    def auto_only(self) -> bool:
        return self.correlation_mode == "AUTO_ONLY"

    @property
    def spws(self) -> list[SpwDef]:
        return [spw for bb in self.basebands for spw in bb]

    @property
    def num_cross_baselines(self) -> int:
        if self.auto_only:
            return 0
        return self.num_antenna * (self.num_antenna - 1) // 2

    @property
    def num_integrations(self) -> int:
        return self.packed_num_time if self.packed_num_time else self.num_subsets

    @property
    def num_apc(self) -> int:
        return 1 if self.auto_only else len(self.apc)

    @property
    def max_bin(self) -> int:
        return max(spw.num_bin for spw in self.spws)

    def overall_spw_idx(self, bb_spw: tuple[int, int]) -> int:
        return sum(len(bb) for bb in self.basebands[: bb_spw[0]]) + bb_spw[1]


def num_auto_values(sd_pols) -> int:
    return 4 if len(sd_pols) == 3 else len(sd_pols)


def cross_code_index(bdef: BDFDef, tim, bl, ospw, bin_idx, apc, chan, pol):
    dims = (
        bdef.num_integrations,
        max(bdef.num_cross_baselines, 1),
        len(bdef.spws),
        bdef.max_bin,
        bdef.num_apc,
        MAX_CHAN,
        4,
    )
    return np.ravel_multi_index(
        np.broadcast_arrays(tim, bl, ospw, bin_idx, apc, chan, pol), dims
    )


def cross_true_value(bdef: BDFDef, tim, bl, ospw, bin_idx, apc, chan, pol):
    idx = cross_code_index(bdef, tim, bl, ospw, bin_idx, apc, chan, pol)
    return ((idx + 1) - 1j * (2 * idx + 1)) / bdef.scale_factor


def auto_true_value(bdef: BDFDef, tim, ant, ospw, bin_idx, chan, k):
    dims = (
        bdef.num_integrations,
        bdef.num_antenna,
        len(bdef.spws),
        bdef.max_bin,
        MAX_CHAN,
        4,
    )
    idx = np.ravel_multi_index(
        np.broadcast_arrays(tim, ant, ospw, bin_idx, chan, k), dims
    )
    return idx + 0.5


def flag_word(tim, row, ospw, k):
    idx = (
        5 * np.asarray(tim)
        + 7 * np.asarray(row)
        + 3 * np.asarray(ospw)
        + 11 * np.asarray(k)
    ) % len(FLAG_WORDS)
    return FLAG_WORDS[idx]


def raw_cross(bdef: BDFDef, tim: int) -> np.ndarray:
    """crossData values of one integration (codes: re, im interleaved)."""
    parts = []
    for bl in range(bdef.num_cross_baselines):
        for ospw, spw in enumerate(bdef.spws):
            b, a, c, p = np.meshgrid(
                np.arange(spw.num_bin),
                np.arange(bdef.num_apc),
                np.arange(spw.nchan),
                np.arange(len(spw.cross_pols)),
                indexing="ij",
            )
            idx = cross_code_index(bdef, tim, bl, ospw, b, a, c, p)
            parts.append(np.stack([idx + 1, -(2 * idx + 1)], axis=-1).ravel())
    return np.concatenate(parts).astype(np.float64)


def raw_auto(bdef: BDFDef, tim: int) -> np.ndarray:
    parts = []
    for ant in range(bdef.num_antenna):
        for ospw, spw in enumerate(bdef.spws):
            b, c, k = np.meshgrid(
                np.arange(spw.num_bin),
                np.arange(spw.nchan),
                np.arange(num_auto_values(spw.sd_pols)),
                indexing="ij",
            )
            parts.append(auto_true_value(bdef, tim, ant, ospw, b, c, k).ravel())
    return np.concatenate(parts)


def _flag_groups(bdef: BDFDef) -> list[tuple[int, SpwDef]]:
    """(flag group id, SPW giving the polarizations) of every group of flags of a row."""
    if bdef.flags_per_baseband:
        return [(100 + bb_idx, bb[0]) for bb_idx, bb in enumerate(bdef.basebands)]
    return list(enumerate(bdef.spws))


def flag_group_id(bdef: BDFDef, bb_spw: tuple[int, int]) -> int:
    if bdef.flags_per_baseband:
        return 100 + bb_spw[0]
    return bdef.overall_spw_idx(bb_spw)


def raw_flags(bdef: BDFDef, tim: int) -> np.ndarray:
    parts = []
    nbl = bdef.num_cross_baselines
    for bl in range(nbl):
        for group, spw in _flag_groups(bdef):
            parts.append(flag_word(tim, bl, group, np.arange(len(spw.cross_pols))))
    for ant in range(bdef.num_antenna):
        for group, spw in _flag_groups(bdef):
            parts.append(flag_word(tim, nbl + ant, group, np.arange(len(spw.sd_pols))))
    return np.concatenate(parts).astype(np.int32)


def _stored_cross(bdef: BDFDef, codes: np.ndarray, endian: str) -> np.ndarray:
    kind = CROSS_TYPES[bdef.cross_type]
    dtype = np.dtype(endian + kind)
    if kind.startswith("i"):
        info = np.iinfo(dtype)
        assert codes.min() >= info.min and codes.max() <= info.max
        return codes.astype(dtype)
    return (codes / bdef.scale_factor).astype(dtype)


def _axes(bdef: BDFDef, component: str) -> str:
    tim = "TIM " if bdef.packed_num_time else ""
    binax = "BIN " if bdef.max_bin > 1 else ""
    if component == "crossData":
        apc = "APC " if bdef.num_apc > 1 else ""
        return f"{tim}BAL BAB SPW {binax}{apc}SPP POL"
    if component == "autoData":
        return f"{tim}ANT BAB SPW {binax}SPP POL"
    spw = "" if bdef.flags_per_baseband else "SPW "
    if bdef.auto_only:
        return f"{tim}ANT BAB {spw}POL"
    return f"{tim}BAL ANT BAB {spw}POL"


def write_bdf(bdef: BDFDef, path: str) -> str:
    """Writes the BDF to path and returns path."""
    endian = ">" if bdef.big_endian else "<"
    ntim = bdef.packed_num_time or 1
    if bdef.packed_num_time:
        subsets = [list(range(bdef.packed_num_time))]
    else:
        subsets = [[idx] for idx in range(bdef.num_subsets)]

    bb_xml = ""
    for bb_idx, bb in enumerate(bdef.basebands):
        bb_xml += f'<baseband name="BB_{bb_idx + 1}">'
        for spw_idx, spw in enumerate(bb):
            attrs = (
                f'sw="{spw_idx + 1}" swbb="BB_{bb_idx + 1}_SW-{spw_idx + 1}" '
                f'numSpectralPoint="{spw.nchan}" numBin="{spw.num_bin}" sideband="USB"'
            )
            if not bdef.auto_only:
                attrs += (
                    f' crossPolProducts="{" ".join(spw.cross_pols)}"'
                    f' scaleFactor="{bdef.scale_factor!r}"'
                )
            attrs += f' sdPolProducts="{" ".join(spw.sd_pols)}"'
            bb_xml += f"<spectralWindow {attrs}/>"
        bb_xml += "</baseband>"

    struct = bb_xml
    if bdef.with_flags:
        struct += (
            f'<flags size="{raw_flags(bdef, 0).size * ntim}" '
            f'axes="{_axes(bdef, "flags")}"/>'
        )
    if not bdef.auto_only:
        struct += (
            f'<crossData size="{raw_cross(bdef, 0).size * ntim}" '
            f'axes="{_axes(bdef, "crossData")}"/>'
        )
    struct += (
        f'<autoData size="{raw_auto(bdef, 0).size * ntim}" '
        f'axes="{_axes(bdef, "autoData")}" normalized="false"/>'
    )
    apc_attr = "" if bdef.auto_only else f' apc="{" ".join(bdef.apc)}"'
    byte_order = "Big_Endian" if bdef.big_endian else "Little_Endian"
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sdmDataHeader xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'byteOrder="{byte_order}" schemaVersion="2" projectPath="1/1/1/">\n'
        "<startTime>5196716800000000000</startTime>\n"
        '<dataOID xlink:href="uid://A002/X1/X1" xlink:title="synthetic"/>\n'
        + (
            f"<numTime>{bdef.packed_num_time}</numTime>\n"
            if bdef.packed_num_time
            else "<dimensionality>1</dimensionality>\n"
        )
        + '<execBlock xlink:href="uid://A002/X1/X0"/>\n'
        f"<numAntenna>{bdef.num_antenna}</numAntenna>\n"
        f"<correlationMode>{bdef.correlation_mode}</correlationMode>\n"
        "<spectralResolution>FULL_RESOLUTION</spectralResolution>\n"
        f"<processorType>{bdef.processor_type}</processorType>\n"
        f"<dataStruct{apc_attr}>{struct}</dataStruct>\n"
        "</sdmDataHeader>\n"
    )

    out = bytearray()
    out += b"MIME-Version: 1.0\n"
    out += (
        f'Content-Type: multipart/mixed; boundary="{_B1}"; type="text/xml"\n'.encode()
    )
    out += b"Content-Description: Correlator\n"
    out += b"Content-Location: uid://A002/X1/X1/\n\n"
    out += f"--{_B1}\n".encode()
    out += b'Content-Type: text/xml; charset="utf-8"\n'
    out += b"Content-Location: sdmDataHeader.xml\n\n"
    out += header.encode()

    for sub_idx, tims in enumerate(subsets):
        missing = bdef.missing_components.get(sub_idx, set())
        project_path = f"1/1/1/{sub_idx + 1}/"
        components = []
        if bdef.with_flags and "flags" not in missing:
            flags = np.concatenate([raw_flags(bdef, tim) for tim in tims])
            components.append(("flags", flags.astype(endian + "i4")))
        if not bdef.auto_only and "crossData" not in missing:
            codes = np.concatenate([raw_cross(bdef, tim) for tim in tims])
            components.append(("crossData", _stored_cross(bdef, codes, endian)))
        if "autoData" not in missing:
            auto = np.concatenate([raw_auto(bdef, tim) for tim in tims])
            components.append(("autoData", auto.astype(endian + "f4")))

        out += f"--{_B1}\n".encode()
        out += (
            f'Content-Type: multipart/related; boundary="{_B2}"; type="text/xml"\n'
        ).encode()
        out += b"Content-Description: Data and metadata subset\n"
        out += f"--{_B2}\n".encode()
        out += b'Content-Type: text/xml; charset="utf-8"\n'
        out += f"Content-Location: {project_path}desc.xml\n\n".encode()
        sub_xml = (
            '<sdmDataSubsetHeader xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'projectPath="{project_path}">\n'
            f"<schedulePeriodTime><time>{5196716800000000000 + sub_idx * 10**9}</time>"
            f"<interval>{10**9}</interval></schedulePeriodTime>\n"
            '<dataStruct ref="sdmDataHeader"/>\n'
        )
        for name, _ in components:
            type_attr = f' type="{bdef.cross_type}"' if name == "crossData" else ""
            sub_xml += f'<{name} xlink:href="{project_path}{name}.bin"{type_attr}/>\n'
        sub_xml += "</sdmDataSubsetHeader>\n"
        out += sub_xml.encode()
        for name, arr in components:
            out += f"--{_B2}\n".encode()
            out += b"Content-Type: binary/octet-stream\n"
            out += f"Content-Location: {project_path}{name}.bin\n\n".encode()
            out += np.ascontiguousarray(arr).tobytes()
            out += b"\n"
        out += f"--{_B2}--\n".encode()
    out += f"--{_B1}--\n".encode()

    with open(path, "wb") as bdf_file:
        bdf_file.write(bytes(out))
    return path


def open_bdf(path: str) -> tuple[pyasdm.bdf.BDFReader, dict]:
    """Opens a BDF and returns the reader and the BDF description dict."""
    reader = pyasdm.bdf.BDFReader()
    reader.open(path)
    header = reader.getHeader()
    bdf_descr = {
        "dimensionality": header.getDimensionality(),
        "num_time": header.getNumTime(),
        "processor_type": header.getProcessorType(),
        "binary_types": header.getBinaryTypes(),
        "correlation_mode": header.getCorrelationMode(),
        "apc": header.getAPClist(),
        "num_antenna": header.getNumAntenna(),
        "basebands": header.getBasebandsList(),
    }
    return reader, bdf_descr


def loaded_apc_index(bdef: BDFDef) -> int:
    """
    Index of the APC whose crossData are loaded: AP_UNCORRECTED, or the only APC
    when there is one (whatever it is).
    """
    if bdef.auto_only or len(bdef.apc) <= 1:
        return 0
    return list(bdef.apc).index("AP_UNCORRECTED")


def expected_visibility(bdef: BDFDef, bb_spw: tuple[int, int], tims=None) -> np.ndarray:
    """
    True (time, baseline, frequency, polarization) visibilities of one SPW:
    cross baselines followed by autos (antennas for AUTO_ONLY), AP_UNCORRECTED
    data (or the only APC), full-pol autos as [XX, XY, conj(XY), YY] ([XX, XY, YY]
    for AUTO_ONLY).
    """
    tims = range(bdef.num_integrations) if tims is None else tims
    spw = bdef.basebands[bb_spw[0]][bb_spw[1]]
    ospw = bdef.overall_spw_idx(bb_spw)
    npol = len(spw.sd_pols) if bdef.auto_only else len(spw.cross_pols)
    nbl = bdef.num_cross_baselines
    apc = loaded_apc_index(bdef)
    out = np.zeros((len(tims), nbl + bdef.num_antenna, spw.nchan, npol), complex)
    chan = np.arange(spw.nchan)[:, None]
    for tidx, tim in enumerate(tims):
        for bl in range(nbl):
            out[tidx, bl] = cross_true_value(
                bdef, tim, bl, ospw, 0, apc, chan, np.arange(npol)[None, :]
            )
        for ant in range(bdef.num_antenna):
            vals = auto_true_value(
                bdef,
                tim,
                ant,
                ospw,
                0,
                chan,
                np.arange(num_auto_values(spw.sd_pols))[None, :],
            )
            if len(spw.sd_pols) == 3:
                xy = vals[:, 1] + 1j * vals[:, 2]
                if npol == 4:
                    vals = np.stack([vals[:, 0], xy, np.conj(xy), vals[:, 3]], -1)
                else:
                    vals = np.stack([vals[:, 0], xy, vals[:, 3]], -1)
            out[tidx, nbl + ant] = vals
    return out


def expected_flags(bdef: BDFDef, bb_spw: tuple[int, int], tims=None) -> np.ndarray:
    """True (time, baseline, polarization) flags of one SPW (word != 0)."""
    tims = range(bdef.num_integrations) if tims is None else tims
    spw = bdef.basebands[bb_spw[0]][bb_spw[1]]
    ospw = flag_group_id(bdef, bb_spw)
    npol = len(spw.sd_pols) if bdef.auto_only else len(spw.cross_pols)
    nbl = bdef.num_cross_baselines
    out = np.zeros((len(tims), nbl + bdef.num_antenna, npol), bool)
    if not bdef.with_flags:
        return out
    for tidx, tim in enumerate(tims):
        for bl in range(nbl):
            out[tidx, bl] = flag_word(tim, bl, ospw, np.arange(npol)) != 0
        for ant in range(bdef.num_antenna):
            flagged = flag_word(tim, nbl + ant, ospw, np.arange(len(spw.sd_pols))) != 0
            if len(spw.sd_pols) == 3 and npol == 4:
                flagged = flagged[[0, 1, 1, 2]]
            out[tidx, nbl + ant] = flagged
    return out


def write_and_open(bdef: BDFDef, directory, name: str = "bdf.bin"):
    """Writes the BDF in directory and opens it. Returns (path, reader, bdf_descr)."""
    path = write_bdf(bdef, os.path.join(str(directory), name))
    reader, bdf_descr = open_bdf(path)
    return path, reader, bdf_descr


def _dual(nchan: int, num_bin: int = 1) -> SpwDef:
    return SpwDef(nchan, ("XX", "YY"), ("XX", "YY"), num_bin)


def _full(nchan: int) -> SpwDef:
    return SpwDef(nchan, ("XX", "XY", "YX", "YY"), ("XX", "XY", "YY"))


def _sd(nchan: int, sd_pols: tuple[str, ...]) -> SpwDef:
    return SpwDef(nchan, (), sd_pols)


#: Canonical BDF definitions used by the tests, by name
BDF_DEFS = {
    # uneven SPWs per baseband and channels per SPW (float32 data)
    "uneven": lambda: BDFDef([[_dual(8), _dual(1)], [_dual(4)]], num_antenna=4),
    # full polarization, INT32 data with scale factor
    "full_pol_int32": lambda: BDFDef(
        [[_full(4)], [_full(2)]], cross_type="INT32_TYPE", scale_factor=4.0
    ),
    # INT16 data, different polarizations per SPW
    "mixed_pols_int16": lambda: BDFDef(
        [[_dual(3), SpwDef(2, ("XX",), ("XX",))]],
        cross_type="INT16_TYPE",
        scale_factor=2.0,
    ),
    # 2 APCs, AP_UNCORRECTED second
    "two_apc": lambda: BDFDef(
        [[_dual(4)], [_dual(3)]],
        cross_type="INT32_TYPE",
        apc=("AP_CORRECTED", "AP_UNCORRECTED"),
    ),
    # big endian data
    "big_endian": lambda: BDFDef(
        [[_dual(5)], [_full(2)]], cross_type="INT32_TYPE", big_endian=True
    ),
    # numBin > 1 in another SPW than the one loaded
    "bins_other_spw": lambda: BDFDef([[_dual(4, num_bin=2)], [_dual(3)]]),
    # flags given per baseband (uneven SPWs per baseband)
    "flags_per_baseband": lambda: BDFDef(
        [[_dual(4)], [_dual(2), _dual(3)]], num_antenna=4, flags_per_baseband=True
    ),
    # single dish
    "auto_only": lambda: BDFDef(
        [[_sd(6, ("XX", "YY")), _sd(2, ("XX", "YY"))]],
        correlation_mode="AUTO_ONLY",
        apc=(),
    ),
    "auto_only_full_pol": lambda: BDFDef(
        [[_sd(4, ("XX", "XY", "YY"))]],
        num_antenna=2,
        correlation_mode="AUTO_ONLY",
        apc=(),
    ),
    # packed (dimensionality 0) WVR-like data: numTime=5 in one subset
    "packed_wvr": lambda: BDFDef(
        [[_sd(4, ("I",))]],
        correlation_mode="AUTO_ONLY",
        processor_type="RADIOMETER",
        apc=(),
        packed_num_time=5,
    ),
}


def all_bb_spws(bdef: BDFDef) -> list[tuple[int, int]]:
    """(baseband index, SPW index) of every SPW of a BDF."""
    return [
        (bb_idx, spw_idx)
        for bb_idx, bb in enumerate(bdef.basebands)
        for spw_idx in range(len(bb))
    ]
