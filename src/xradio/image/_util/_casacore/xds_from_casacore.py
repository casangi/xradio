from astropy import units as u
import copy
import os
from typing import List, Tuple, Union

import dask
import dask.array as da
import toolviper.utils.logger as logger
import numpy as np
import xarray as xr
from astropy import units as u
from casacore import tables
from casacore.images import coordinates

from .common import (
    _active_mask,
    _object_name,
    _open_image_ro,
    _pointing_center,
)
from ..common import (
    _compute_linear_world_values,
    _compute_velocity_values,
    _compute_world_sph_dims,
    _convert_beam_to_rad,
    _default_freq_info,
    _doppler_types,
    _get_unit,
    _image_type,
    _l_m_attr_notes,
)
from ...._utils._casacore.tables import extract_table_attributes, open_table_ro
from xradio._utils.coord_math import _deg_to_rad

"""
def _add_coord_attrs(xds: xr.Dataset, icoords: dict, dir_axes: list) -> xr.Dataset:
    _add_time_attrs(xds, icoords)
    _add_freq_attrs(xds, icoords)
    xds = _add_vel_attrs(xds, icoords)
    xds = _add_lin_attrs(xds, icoords, dir_axes)
    return xds
"""


def _add_lin_attrs(xds, coord_dict, dir_axes):
    for k in coord_dict:
        if k.startswith("linear"):
            ld = coord_dict[k]
            for i in (0, 1):
                meta = {}
                meta["units"] = ld["units"][i]
                meta["crval"] = ld["crval"][i]
                meta["cdelt"] = ld["cdelt"][i]
                xds[dir_axes[i]].attrs = copy.deepcopy(meta)
            break
    return xds


def _add_freq_attrs(xds, coord_dict):
    freq_coord = xds["frequency"]
    meta = {}
    for k in coord_dict:
        if k.startswith("spectral"):
            sd = coord_dict[k]
            # meta["native_type"] = _native_types[sd["nativeType"]]
            meta["rest_frequency"] = {
                "type": "quantity",
                "units": "Hz",
                "value": sd["restfreq"],
            }
            # meta["restfreqs"] = {'type': 'quantity', 'units': 'Hz', 'value': list(sd["restfreqs"])}
            meta["type"] = "frequency"
            meta["units"] = sd["unit"]
            meta["frame"] = sd["system"]
            meta["wave_unit"] = sd["waveUnit"]
            meta["crval"] = sd["wcs"]["crval"]
            meta["cdelt"] = sd["wcs"]["cdelt"]
    if not meta:
        # this is the default frequency information CASA creates
        meta = _default_freq_info()
    freq_coord.attrs = meta


def _add_mask(
    xds: xr.Dataset, name: str, ary: Union[np.ndarray, da.array], dimorder: list
) -> xr.Dataset:
    xda = xr.DataArray(ary, dims=dimorder)
    # True pixels are good in numpy masked arrays
    xda = da.logical_not(xda)
    xda.attrs["image_type"] = "Mask"
    xda = xda.rename(name)
    xds[xda.name] = xda
    return xds


def _add_sky_or_aperture(
    xds: xr.Dataset,
    ary: Union[np.ndarray, da.array],
    dimorder: list,
    img_full_path: str,
    has_sph_dims: bool,
) -> xr.Dataset:
    xda = xr.DataArray(ary, dims=dimorder).astype(ary.dtype)
    with _open_image_ro(img_full_path) as casa_image:
        image_type = casa_image.info()["imageinfo"]["imagetype"]
        unit = casa_image.unit()
    xda.attrs[_image_type] = image_type
    xda.attrs["units"] = unit
    name = "SKY" if has_sph_dims else "APERTURE"
    xda = xda.rename(name)
    xds[xda.name] = xda
    return xds


def _get_time_format(value: float, unit: str) -> str:
    if value >= 40000 and value <= 100000 and unit == "d":
        return "MJD"
    else:
        return ""


def _add_time_attrs(xds: xr.Dataset, coord_dict: dict) -> xr.Dataset:
    # time_coord = xds['time']
    meta = {}
    meta["type"] = "time"
    meta["scale"] = coord_dict["obsdate"]["refer"]
    meta["units"] = coord_dict["obsdate"]["m0"]["unit"]
    meta["format"] = _get_time_format(xds["time"][0], meta["units"])
    xds["time"].attrs = copy.deepcopy(meta)
    # xds['time'] = time_coord
    return xds


def _add_vel_attrs(xds: xr.Dataset, coord_dict: dict) -> xr.Dataset:
    vel_coord = xds["velocity"]
    meta = {"units": "m/s"}
    for k in coord_dict:
        if k.startswith("spectral"):
            sd = coord_dict[k]
            meta["doppler_type"] = _doppler_types[sd["velType"]]
            break
    if not meta:
        meta["doppler_type"] = _doppler_types[0]
    meta["type"] = "doppler"
    vel_coord.attrs = copy.deepcopy(meta)
    xds["velocity"] = vel_coord
    return xds


def _casa_image_to_xds_attrs(img_full_path: str, history: bool = True) -> dict:
    """
    Get the xds level attribut/es as a python dictionary
    """
    with _open_image_ro(img_full_path) as casa_image:
        meta_dict = casa_image.info()
    coord_dict = copy.deepcopy(meta_dict["coordinates"])
    attrs = {}
    dir_key = None
    for k in coord_dict.keys():
        if k.startswith("direction"):
            dir_key = k
            break
    if dir_key:
        # shared direction coordinate attributes
        coord_dir_dict = copy.deepcopy(coord_dict[dir_key])
        system = "system"
        if system not in coord_dir_dict:
            raise RuntimeError("No direction reference frame found")
        casa_system = coord_dir_dict[system]
        ap_system, ap_equinox = _convert_direction_system(casa_system, "native")
        dir_dict = {}
        dir_dict["reference"] = {
            "frame": ap_system,
            "type": "sky_coord",
            "equinox": ap_equinox if ap_equinox else None,
            "value": [0.0, 0.0],
            "units": ["rad", "rad"],
        }
        for i in range(2):
            unit = u.Unit(_get_unit(coord_dir_dict["units"][i]))
            q = coord_dir_dict["crval"][i] * unit
            x = q.to("rad")
            dir_dict["reference"]["value"][i] = x.value
        k = "latpole"
        if k in coord_dir_dict:
            for j in (k, "longpole"):
                dir_dict[j] = {
                    "value": coord_dir_dict[j] * _deg_to_rad,
                    "units": "rad",
                    "type": "quantity",
                }
        for j in ("pc", "projection_parameters", "projection"):
            if j in coord_dir_dict:
                dir_dict[j] = coord_dir_dict[j]
        attrs["direction"] = dir_dict
    attrs["telescope"] = {}
    telescope = attrs["telescope"]
    attrs["obsdate"] = {"type": "time"}
    obsdate = attrs["obsdate"]
    attrs[_pointing_center] = coord_dict["pointingcenter"].copy()
    for k in ("observer", "obsdate", "telescope", "telescopeposition"):
        if k.startswith("telescope"):
            if k == "telescope":
                telescope["name"] = coord_dict[k]
            elif k in coord_dict:
                telescope["position"] = coord_dict[k]
                telescope["position"]["ellipsoid"] = telescope["position"]["refer"]
                if telescope["position"]["refer"] == "ITRF":
                    telescope["position"]["ellipsoid"] = "GRS80"
                telescope["position"]["units"] = [
                    telescope["position"]["m0"]["unit"],
                    telescope["position"]["m1"]["unit"],
                    telescope["position"]["m2"]["unit"],
                ]
                telescope["position"]["value"] = [
                    telescope["position"]["m0"]["value"],
                    telescope["position"]["m1"]["value"],
                    telescope["position"]["m2"]["value"],
                ]
                del (
                    telescope["position"]["refer"],
                    telescope["position"]["m0"],
                    telescope["position"]["m1"],
                    telescope["position"]["m2"],
                )
        elif k == "obsdate":
            obsdate["scale"] = coord_dict[k]["refer"]
            obsdate["units"] = coord_dict[k]["m0"]["unit"]
            obsdate["value"] = coord_dict[k]["m0"]["value"]
            obsdate["format"] = _get_time_format(obsdate["value"], obsdate["units"])
            obsdate["type"] = "time"
        else:
            attrs[k] = coord_dict[k] if k in coord_dict else ""
    imageinfo = meta_dict["imageinfo"]
    obj = "objectname"
    attrs[_object_name] = imageinfo[obj] if obj in imageinfo else ""
    attrs["beam"] = _get_beam(imageinfo)
    attrs["user"] = meta_dict["miscinfo"]
    defmask = "Image_defaultmask"
    with open_table_ro(img_full_path) as casa_table:
        attrs[_active_mask] = (
            casa_table.getkeyword(defmask)
            if defmask in casa_table.keywordnames()
            else None
        )
    attrs["description"] = None
    # if also loading history, put it as another xds in the attrs
    if history:
        htable = os.sep.join([img_full_path, "logtable"])
        if os.path.isdir(htable):
            attrs["history"] = read_generic_table(htable)
        else:
            logger.warning(
                f"Unable to find history table {htable}. History will not be included"
            )
    return copy.deepcopy(attrs)


def _casa_image_to_xds_coords(
    img_full_path: str, verbose: bool, do_sky_coords: bool
) -> dict:
    """
    TODO: complete documentation
    Create an xds without any pixel data from metadata from the specified CASA image
    """
    attrs = {}
    # casa_image = images.image(img_full_path)
    with _open_image_ro(img_full_path) as casa_image:
        # shape list is the reverse of the actual image shape
        shape = casa_image.shape()[::-1]
        meta_dict = casa_image.info()
        csys = casa_image.coordinates()
    axis_names = _flatten_list(csys.get_axes())[::-1]
    coord_dict = meta_dict["coordinates"]
    attrs["icoords"] = coord_dict
    diraxes = [
        aa.lower().replace(" ", "_")
        for cc in coord_dict.items()
        if (cc[0][:-1] in ["direction", "linear"]) and len(cc[1]["axes"]) >= 2
        for aa in cc[1]["axes"]
    ]
    attrs["dir_axes"] = diraxes
    dimmap = _get_dimmap(coord_dict, verbose)
    attrs["dimmap"] = dimmap
    sphr_dims = (
        [dimmap["l"], dimmap["m"]] if ("l" in dimmap) and ("m" in dimmap) else []
    )
    attrs["sphr_dims"] = sphr_dims
    coords = {}
    coord_attrs = {}
    (coords["time"], coord_attrs["time"]) = _get_time_values_attrs(coord_dict)
    (coords["polarization"], coord_attrs["polarization"]) = _get_pol_values_attrs(
        coord_dict
    )
    (coords["frequency"], coord_attrs["frequency"]) = _get_freq_values_attrs(
        csys, shape
    )
    (velocity_vals, coord_attrs["velocity"]) = _get_velocity_values_attrs(
        coord_dict, coords["frequency"]
    )
    coords["velocity"] = (["frequency"], velocity_vals)
    if len(sphr_dims) > 0:
        crpix = _flatten_list(csys.get_referencepixel())[::-1]
        inc = _flatten_list(csys.get_increment())[::-1]
        unit = _flatten_list(csys.get_unit())[::-1]
        attr_note = _l_m_attr_notes()
        for c in ["l", "m"]:
            idx = dimmap[c]
            delta = ((inc[idx]) * u.Unit(_get_unit(unit[idx]))).to("rad").value
            coords[c] = _compute_linear_world_values(
                naxis=shape[idx], crval=0.0, crpix=crpix[idx], cdelt=delta
            )
            coord_attrs[c] = {
                "crval": 0.0,
                "cdelt": delta,
                "units": "rad",
                "type": "quantity",
                "note": attr_note[c],
            }
        if do_sky_coords:
            for k in coord_dict.keys():
                if k.startswith("direction"):
                    dc = coordinates.directioncoordinate(coord_dict[k])
                    break
            crval = _flatten_list(csys.get_referencevalue())[::-1]
            pick = lambda my_list: [my_list[i] for i in sphr_dims]
            my_ret = _compute_world_sph_dims(
                projection=dc.get_projection(),
                shape=pick(shape),
                ctype=diraxes,
                crval=pick(crval),
                crpix=pick(crpix),
                cdelt=pick(inc),
                cunit=pick(unit),
            )
            for i in [0, 1]:
                axis_name = my_ret["axis_name"][i]
                coords[axis_name] = (["l", "m"], my_ret["value"][i])
                coord_attrs[axis_name] = {}
    else:
        # Fourier image
        ret = _get_uv_values_attrs(coord_dict, axis_names, shape)
        for z in ["u", "v"]:
            coords[z], coord_attrs[z] = ret[z]
    attrs["shape"] = shape
    xds = xr.Dataset(coords=coords)
    for c in coord_attrs.keys():
        xds[c].attrs = coord_attrs[c]
    attrs["xds"] = xds
    return attrs


def _convert_direction_system(
    casa_system: str, which: str, verbose: bool = True
) -> tuple:
    if casa_system == "J2000":
        if verbose:
            logger.info(
                f"J2000 found as {which} reference frame in CASA image "
                'This corresponds to FK5(equinox="J2000") in astropy. '
                "Metadata will be written appropriately"
            )
        return ("FK5", "J2000.0")
    elif casa_system == "B1950":
        if verbose:
            logger.info(
                f"B1950 found as {which} reference frame in CASA image "
                'This corresponds to FK4(equinox="B1950") in astropy. '
                "Metadata will be written appropriately"
            )
        return ("FK4", "B1950.0")
    elif casa_system in ("GALACTIC", "ICRS"):
        return (casa_system.lower(), None)
    else:
        raise Exception(
            f"astropy does not support frame {casa_system} and this "
            "application does not support converting it to "
            "something astropy does support. You can try to regridding "
            "it to ICRS, GALACTIC, J2000, or B1950 in CASA and then "
            "re-run this application on the regridded image"
        )


def _flatten_list(list_of_lists: list) -> list:
    flat = []
    for x in list_of_lists:
        if type(x) == list or type(x) == np.ndarray:
            flat.extend(_flatten_list(x))
        else:
            flat.append(x)
    return flat


def _get_beam(imageinfo: dict):
    """Returns None if no beam. Multiple beams are handled elsewhere"""
    k = "restoringbeam"
    if k in imageinfo and "major" in imageinfo[k]:
        return _convert_beam_to_rad(imageinfo[k])
    return None


def _get_chunk_list(
    chunks: dict, coords: list, image_shape: Union[list, tuple]
) -> tuple:
    ret_list = list(image_shape)
    axis = 0
    for c in coords:
        if c == "direction" or c == "linear":
            lm = ("l", "m")
            uv = ("u", "v")
            for i in (0, 1):
                for k in (lm[i], uv[i]):
                    if k in chunks:
                        ret_list[axis] = chunks[k]
                        break
                # add an axis because direction has 2 axes
                if i == 0:
                    axis += 1
        elif c == "spectral":
            if "frequency" in chunks:
                ret_list[axis] = chunks["frequency"]
        elif c == "stokes":
            if "polarization" in chunks:
                ret_list[axis] = chunks["polarization"]
        else:
            raise Exception(f"Unhandled coordinate type {c}")
        axis += 1
    return tuple(ret_list)


def _get_dimmap(coords: list, verbose: bool = False) -> dict:
    # example of dimmap:
    # [('direction0', 0), ('direction1', 1), ('spectral0', 2), ('stokes0', 3)]
    dimmap: list[tuple] = [
        (coord[:-1] + str(ii), ci)
        for coord in coords
        if coord[:-1] in ["direction", "stokes", "spectral", "linear"]
        for ii, ci in enumerate(coords["pixelmap%s" % coord[-1]])
    ]
    if verbose:
        print(f"dimmap: {dimmap}")
    # example of dimmap after next statment
    # [('l', 0), ('m', 1), ('chan', 2), ('polarization', 3)]
    dimmap = [
        (
            rr[0]
            .replace("stokes0", "polarization")
            .replace("spectral0", "chan")
            .replace("direction0", "l")
            .replace("direction1", "m")
            .replace("linear0", "u")
            .replace("linear1", "v"),
            rr[1],
        )
        for rr in dimmap
    ]
    if verbose:
        print(f"dimmap: {dimmap}")
    if ("linear0" in np.vstack(dimmap)[:, 0]) and (
        "linear1" in np.vstack(dimmap)[:, 0]
    ):
        dimmap = [
            (rr[0].replace("linear0", "l").replace("linear1", "m"), rr[1])
            for rr in dimmap
        ]
    dimmap = [
        (rr[0].replace("linear0", "component"), rr[1]) for rr in dimmap if rr[1] >= 0
    ]
    # conversion to dict, example dimmap after this statement
    # dimmap: {'l': 0, 'm': 1, 'chan': 2, 'polarization': 3}
    dimmap = dict(
        [
            (
                (diraxes[int(rr[0][-1])], rr[1])
                if rr[0].startswith("linear") or rr[0].startswith("direction")
                else rr
            )
            for rr in dimmap
        ]
    )
    if verbose:
        print(f"dimmap: {dimmap}")
    return dimmap


def _get_freq_values(coords: coordinates.coordinatesystem, shape: tuple) -> list:
    idx = _get_image_axis_order(coords)[::-1].index("Frequency")
    if idx >= 0:
        coord_dict = coords.dict()
        for k in coord_dict:
            if k.startswith("spectral"):
                wcs = coord_dict[k]["wcs"]
                return _compute_linear_world_values(
                    naxis=shape[idx],
                    crval=wcs["crval"],
                    crpix=wcs["crpix"],
                    cdelt=wcs["cdelt"],
                )
    else:
        return [1420e6]


def _get_freq_values_attrs(
    casa_coords: coordinates.coordinatesystem, shape: tuple
) -> Tuple[List[float], dict]:
    values = None
    attrs = {}
    idx = _get_image_axis_order(casa_coords)[::-1].index("Frequency")
    if idx >= 0:
        casa_coord_dict = casa_coords.dict()
        for k in casa_coord_dict:
            if k.startswith("spectral"):
                sd = casa_coord_dict[k]
                wcs = sd["wcs"]
                values = _compute_linear_world_values(
                    naxis=shape[idx],
                    crval=wcs["crval"],
                    crpix=wcs["crpix"],
                    cdelt=wcs["cdelt"],
                )
                attrs["rest_frequency"] = {
                    "type": "quantity",
                    "units": "Hz",
                    "value": sd["restfreq"],
                }
                attrs["type"] = "frequency"
                attrs["units"] = sd["unit"]
                attrs["frame"] = sd["system"]
                attrs["wave_unit"] = sd["waveUnit"]
                attrs["crval"] = sd["wcs"]["crval"]
                attrs["cdelt"] = sd["wcs"]["cdelt"]
                attrs = copy.deepcopy(attrs)
                break
    else:
        values = [1420e6]
        # this is the default frequency information CASA creates
        attrs = _default_freq_info()
    return (values, attrs)


def _get_image_axis_order(coords: coordinates.coordinatesystem) -> list:
    """
    get the *reverse* order of image axes
    """
    axis_names = coords.get_axes()[::-1]
    ncoords = len(axis_names)
    csys = coords.dict()
    ordered = len(_flatten_list(axis_names)) * [""]
    for i in range(ncoords - 1, -1, -1):
        axes = csys["pixelmap" + str(i)]
        if len(axes) == 1:
            ordered[axes[0]] = axis_names[i]
        elif len(axes) == 2:
            ordered[axes[0]] = axis_names[i][1]
            ordered[axes[1]] = axis_names[i][0]
    ordered = _flatten_list(ordered)[::-1]
    return ordered


def _get_image_dim_order(coords: coordinates.coordinatesystem) -> list:
    """
    Get the xds dim order of the input image. The returned list is in casacore
    order, which is the reverse of the actual image dimension order
    """
    flat = _get_image_axis_order(coords)
    ret = []
    for axis in flat:
        b = axis.lower()
        if b.startswith("right") or b.startswith("uu"):
            ret.append("l")
        elif b.startswith("dec") or b.startswith("vv"):
            ret.append("m")
        elif b.startswith("frequency"):
            ret.append("frequency")
        elif b.startswith("stok"):
            ret.append("polarization")
        else:
            raise Exception(f"Unhandled axis name {c}")
    return ret


def _get_mask_names(infile: str) -> list:
    t = tables.table(infile)
    tb_tool = tables.table(
        infile, readonly=True, lockoptions={"option": "usernoread"}, ack=False
    )
    mymasks = t.getkeyword("masks") if "masks" in t.keywordnames() else []
    t.close()
    return mymasks


def _get_multibeam(imageinfo: dict) -> Union[np.ndarray, None]:
    """Returns None if the image does not have multiple (per-plane) beams"""
    p = "perplanebeams"
    if p not in imageinfo:
        return None
    beam = imageinfo[p]
    nchan = beam["nChannels"]
    npol = beam["nStokes"]
    beam_array = np.zeros([1, npol, nchan, 3])
    for c in range(nchan):
        for p in range(npol):
            k = nchan * p + c
            b = beam["*" + str(k)]
            beam_dict = _convert_beam_to_rad(b)
            beam_array[0][p][c][0] = beam_dict["major"]["value"]
            beam_array[0][p][c][1] = beam_dict["minor"]["value"]
            beam_array[0][p][c][2] = beam_dict["pa"]["value"]
    return beam_array


def _get_persistent_block(
    infile: str,
    shapes: tuple,
    starts: tuple,
    dimorder: list,
    transpose_list: list,
    new_axes: list,
) -> xr.DataArray:
    block = _read_image_chunk(infile, shapes, starts)
    block = np.expand_dims(block, new_axes)
    block = block.transpose(transpose_list)
    block = xr.DataArray(block, dims=dimorder)
    return block


def _get_pol_values_attrs(casa_coord_dict: dict) -> Tuple[List[str], dict]:
    values = None
    for k in casa_coord_dict:
        if k.startswith("stokes"):
            values = casa_coord_dict[k]["stokes"]
    if values is None:
        values = ["I"]
    return (values, {})


def _get_pol_values(coord_dict: dict) -> List[str]:
    for k in coord_dict:
        if k.startswith("stokes"):
            return coord_dict[k]["stokes"]
    return ["I"]


def _get_starts_shapes_slices(
    blockdes: dict, coords: coordinates.coordinatesystem, cshape: list
) -> tuple:
    img_dim_order = _get_image_dim_order(coords)
    starts = []
    shapes = []
    slices = {}
    for i, dim in enumerate(img_dim_order):
        if dim not in ["polarization", "frequency", "l", "m", "u", "v"]:
            raise RuntimeError(f"Unsupported dimension {dim}")
        if dim in blockdes:
            extent = blockdes[dim]
            if isinstance(extent, int):
                starts.append(extent)
                shapes.append(extent + 1)
                slices[dim] = slice(extent, extent + 1)
            elif isinstance(extent, slice):
                starts.append(extent.start)
                shapes.append(extent.stop - extent.start)
                slices[dim] = slice(extent.start, extent.stop)
            else:
                raise Exception(f"Unhandled extent type {type(extent)}")
        else:
            starts.append(0)
            shapes.append(cshape[i])
    return starts, shapes, slices


def _get_time_values_attrs(cimage_coord_dict: dict) -> Tuple[List[float], dict]:
    attrs = {}
    attrs["type"] = "time"
    attrs["scale"] = cimage_coord_dict["obsdate"]["refer"]
    unit = cimage_coord_dict["obsdate"]["m0"]["unit"]
    attrs["units"] = unit
    time_val = cimage_coord_dict["obsdate"]["m0"]["value"]
    attrs["format"] = _get_time_format(time_val, unit)
    return ([time_val], copy.deepcopy(attrs))


def _get_time_values(coord_dict):
    return [coord_dict["obsdate"]["m0"]["value"]]


def _get_transpose_list(coords: coordinates.coordinatesystem) -> list:
    flat = _get_image_axis_order(coords)
    transpose_list = 5 * [-1]
    # time axis
    transpose_list[0] = 4
    new_axes = [4]
    last_axis = 3
    not_covered = ["l", "m", "u", "v", "s", "f"]
    csys = coords.dict()
    for i, c in enumerate(flat):
        b = c.lower()
        if b.startswith("right") or b.startswith("uu"):
            transpose_list[3] = i
            # transpose_list[3] = csys['pixelmap0'][0]
            not_covered.remove("l")
            not_covered.remove("u")
        elif b.startswith("dec") or b.startswith("vv"):
            transpose_list[4] = i
            # transpose_list[4] = csys['pixelmap0'][1]
            not_covered.remove("m")
            not_covered.remove("v")
        elif b.startswith("frequency"):
            # transpose_list[2] = csys['pixelmap1'][0]
            transpose_list[2] = i
            not_covered.remove("f")
        elif b.startswith("stok"):
            transpose_list[1] = i
            # transpose_list[1] = csys['pixelmap2'][0]
            not_covered.remove("s")
        else:
            raise Exception(f"Unhandled axis name {c}")
    h = {"l": 3, "m": 4, "u": 3, "v": 4, "f": 2, "s": 1}
    for p in not_covered:
        transpose_list[h[p]] = last_axis
        new_axes.append(last_axis)
        last_axis -= 1
    new_axes.sort()
    if transpose_list.count(-1) > 0:
        raise Exception(f"Logic error: axes {axes}, transpose_list {transpose_list}")
    return transpose_list, new_axes


def _get_uv_values(
    coord_dict: dict, axis_names: List[str], shape: Tuple[int]
) -> Tuple[List[float]]:
    for i, axis in enumerate(["UU", "VV"]):
        idx = axis_names.index(axis)
        if idx >= 0:
            for k in coord_dict:
                cdict = coord_dict[k]
                if k.startswith("linear"):
                    z = []
                    crpix = cdict["crpix"][i]
                    crval = cdict["crval"][i]
                    cdelt = cdict["cdelt"][i]
                    for i in range(shape[idx]):
                        f = (i - crpix) * cdelt + crval
                        z.append(f)
                    if axis == "UU":
                        u = z
                    else:
                        v = z
    return u, v


def _get_uv_values_attrs(
    casa_coord_dict: dict, axis_names: List[str], shape: Tuple[int]
) -> dict:
    ret = {}
    for i, axis in enumerate(["UU", "VV"]):
        idx = axis_names.index(axis)
        if idx >= 0:
            for k in casa_coord_dict:
                cdict = casa_coord_dict[k]
                if k.startswith("linear"):
                    crpix = cdict["crpix"][i]
                    crval = cdict["crval"][i]
                    cdelt = cdict["cdelt"][i]
                    attrs = {
                        "crval": crval,
                        "cdelt": cdelt,
                        "units": cdict["units"][i],
                        "type": "quantity",
                    }
                    z = _compute_linear_world_values(
                        naxis=shape[idx],
                        crpix=crpix,
                        crval=crval,
                        cdelt=cdelt,
                    )
                    name = "u" if axis == "UU" else "v"
                    ret[name] = (z, attrs)
    return ret


def _get_velocity_values(coord_dict: dict, freq_values: list) -> list:
    restfreq = 1420405751.786
    for k in coord_dict:
        if k.startswith("spectral"):
            restfreq = coord_dict[k]["restfreq"]
            break
    return _compute_velocity_values(
        restfreq=restfreq, freq_values=freq_values, doppler="RADIO"
    )


def _get_velocity_values_attrs(
    coord_dict: dict, freq_values: List[float]
) -> Tuple[List[float], dict]:
    restfreq = 1420405751.786
    attrs = {}
    for k in coord_dict:
        if k.startswith("spectral"):
            sd = coord_dict[k]
            restfreq = sd["restfreq"]
            attrs["doppler_type"] = _doppler_types[sd["velType"]]
            break
    if not attrs:
        attrs["doppler_type"] = _doppler_types[0]

    attrs["units"] = "m/s"
    attrs["type"] = "doppler"
    return (
        _compute_velocity_values(
            restfreq=restfreq, freq_values=freq_values, doppler="RADIO"
        ),
        copy.deepcopy(attrs),
    )


def _multibeam_array(
    xds: xr.Dataset, img_full_path: str, as_dask_array: bool
) -> Union[xr.DataArray, None]:
    """This should only be called after the xds.beam attr has been set"""
    if xds.attrs["beam"] is None:
        # the image may have multiple beams
        with _open_image_ro(img_full_path) as casa_image:
            imageinfo = casa_image.info()["imageinfo"]
        mb = _get_multibeam(imageinfo)
        if mb is not None:
            # multiple beams are stored as a data varialbe, so remove
            # the beam xds attr
            del xds.attrs["beam"]
            if as_dask_array:
                mb = da.array(mb)
            xdb = xr.DataArray(
                mb, dims=["time", "polarization", "frequency", "beam_param"]
            )
            xdb = xdb.rename("beam")
            xdb = xdb.assign_coords(beam_param=["major", "minor", "pa"])
            xdb.attrs["units"] = "rad"
            return xdb
    else:
        return None


###########################################################################
##
## read_generic_table() - read casacore table into memory resident xds
##
###########################################################################
def read_generic_table(infile, subtables=False, timecols=None, ignore=None):
    """
    read generic casacore table format to xarray dataset loaded in memory

    Parameters
    ----------
    infile : str
        Input table filename. To read a subtable simply append the subtable folder name under the main table (ie infile = '/path/mytable.tbl/mysubtable')
    subtables : bool
        Whether or not to include subtables underneath the specified table. If true, an attribute called subtables will be added to the returned xds.
        Default False
    timecols : list
        list of column names to convert to numpy datetime format. Default None leaves times as their original casacore format.
    ignore : list
        list of column names to ignore and not try to read. Default None reads all columns

    Returns
    -------
    xarray.core.dataset.Dataset
    """
    if timecols is None:
        timecols = []
    if ignore is None:
        ignore = []

    infile = os.path.expanduser(infile)
    assert os.path.isdir(infile), "invalid input filename to read_generic_table"

    attrs = extract_table_attributes(infile)
    with open_table_ro(infile) as tb_tool:
        if tb_tool.nrows() == 0:
            return xr.Dataset(attrs=attrs)

        dims = ["row"] + ["d%i" % ii for ii in range(1, 20)]
        cols = tb_tool.colnames()
        ctype = dict(
            [
                (col, tb_tool.getcell(col, 0))
                for col in cols
                if ((col not in ignore) and (tb_tool.iscelldefined(col, 0)))
            ]
        )
        mvars, mcoords, xds = {}, {}, xr.Dataset()

        tr = tb_tool.row(ignore, exclude=True)[:]

        # extract data for each col
        for col in ctype.keys():
            if tb_tool.coldatatype(col) == "record":
                continue  # not supported

            try:
                data = np.stack([rr[col] for rr in tr])  # .astype(ctype[col].dtype)
                if isinstance(tr[0][col], dict):
                    data = np.stack(
                        [
                            (
                                rr[col]["array"].reshape(rr[col]["shape"])
                                if len(rr[col]["array"]) > 0
                                else np.array([""])
                            )
                            for rr in tr
                        ]
                    )
            except:
                # sometimes the columns are variable, so we need to standardize to the largest sizes
                if len(np.unique([isinstance(rr[col], dict) for rr in tr])) > 1:
                    continue  # can't deal with this case
                mshape = np.array(max([np.array(rr[col]).shape for rr in tr]))
                try:
                    data = np.stack(
                        [
                            np.pad(
                                (
                                    rr[col]
                                    if len(rr[col]) > 0
                                    else np.array(rr[col]).reshape(
                                        np.arange(len(mshape)) * 0
                                    )
                                ),
                                [(0, ss) for ss in mshape - np.array(rr[col]).shape],
                                "constant",
                                constant_values=np.array([np.nan]).astype(
                                    np.array(ctype[col]).dtype
                                )[0],
                            )
                            for rr in tr
                        ]
                    )
                except:
                    data = []

            if len(data) == 0:
                continue
            if col in timecols:
                convert_time(data)
            if col.endswith("_ID"):
                mcoords[col] = xr.DataArray(
                    data,
                    dims=[
                        "d%i_%i" % (di, ds)
                        for di, ds in enumerate(np.array(data).shape)
                    ],
                )
            else:
                mvars[col] = xr.DataArray(
                    data,
                    dims=[
                        "d%i_%i" % (di, ds)
                        for di, ds in enumerate(np.array(data).shape)
                    ],
                )

        xds = xr.Dataset(mvars, coords=mcoords)
        xds = xds.rename(dict([(dv, dims[di]) for di, dv in enumerate(xds.sizes)]))
        attrs["bad_cols"] = list(
            np.setdiff1d(
                [dv for dv in tb_tool.colnames()],
                [dv for dv in list(xds.data_vars) + list(xds.coords)],
            )
        )

        # if this table has subtables, use a recursive call to store them in subtables attribute
        if subtables:
            stbl_list = sorted(
                [
                    tt
                    for tt in os.listdir(infile)
                    if os.path.isdir(os.path.join(infile, tt))
                    and tables.tableexists(os.path.join(infile, tt))
                ]
            )
            attrs["subtables"] = []
            for ii, subtable in enumerate(stbl_list):
                sxds = read_generic_table(
                    os.path.join(infile, subtable),
                    subtables=subtables,
                    timecols=timecols,
                    ignore=ignore,
                )
                if len(sxds.sizes) != 0:
                    attrs["subtables"] += [(subtable, sxds)]
        xds = xds.assign_attrs(attrs)
    return xds


def _read_image_array(
    infile: str,
    chunks: Union[list, dict],
    mask: str = None,
    verbose: bool = False,
    blc=None,
    trc=None,
) -> dask.array:
    """
    Read an array of image pixels into a dask array. The returned dask array
    will have axes in time, polarization, frequency, l, m order
    If specified, it's the caller's responsibility to ensure the specified blc and
    trc are coincident with chunk corners. If not, there could be performance degradation
    The blc, trc box is inclusive of blc pixel coordinates and exclusive trc pixel
    coordinates, which mimics the behavior of numpy array slicing
    ie blc = [0, 0, 0] and trc = [1023, 1023, 1023] excludes pixel [1023, 1023, 1023]
    and has shape [1023, 1023, 1023]
    blc and trc can be any array type object
    blc and trc are taken wrt the CASA image. Not the casacore image in which they
    would be reversed
    :param infile: Path to the input CASA image
    :type infile: str, required
    :param chunks: The desired dask chunk size.
                   If list, the ordering is based on the ordering of the input
                   axes. So, for example, if the input image has axes RA, Dec,
                   Freq, Stokes and the desired chunking is 40 pixels in RA, 30
                   pixels in Dec, 20 pixels in Freq, and 2 pixels in Stokes,
                   chunks would be specified as [40, 30, 20, 2].
                   If dict, supported optional keys are 'l', 'm', 'frequency', 'polarization',
                   and 'time'. The supported values are positive integers,
                   indicating the length of a chunk on that particular axis. If
                   a key is missing, the associated chunk length along that axis
                   is 1. 'l' represents the longitude like dimension, and 'm'
                   represents the latitude like dimension. For apeature images,
                   'u' may be used in place of 'l', and 'v' in place of 'm'.
    :type chunks: list | dict, required
    :param mask: If specified, this is the associated image mask to read, rather than the actual
                 image pixel values. None means read the image pixel values, not the mask.
    :type mask: str, optional
    :param verbose: Emit debugging messages, default False
    :type verbose: bool
    :param blc: bottom left corner, given in the axes ordering of the input image.None=>The origin
    :type blc: a type that is convertable to a list via list(blc)
    :param trc: top right corner, given in the axes ordering of the input image.None=>image shape - 1
    :type trc: a type that is convertable to a list via list(trc)
    :return: Dask array in time, polarization, frequency, l, m order
    :rtype: dask.array

    """
    img_full_path = os.path.expanduser(infile)
    with _open_image_ro(img_full_path) as casa_image:
        """
        if isinstance(chunks, list):
            mychunks = tuple(chunks)
        elif isinstance(chunks, dict):
        """
        if isinstance(chunks, dict):
            mychunks = _get_chunk_list(
                chunks,
                casa_image.coordinates().get_names()[::-1],
                casa_image.shape()[::-1],
            )
        else:
            raise RuntimeError(
                f"incorrect type {type(chunks)} for parameter chunks. Must be "
                "either tuple or dict"
            )
        # shape list is the reverse of the actual image shape
        cshape = casa_image.shape()
        transpose_list, new_axes = _get_transpose_list(casa_image.coordinates())
        data_type = casa_image.datatype().lower()
    if data_type == "float":
        data_type = np.float32
    elif data_type == "double":
        data_type = np.float64
    elif data_type == "complex":
        data_type = np.complex64
    elif data_type == "dcomplex":
        data_type = np.complex128
    else:
        raise RuntimeError(f"Unhandled data type {data_type}")
    if verbose:
        print(f"cshape: {cshape}")
    if mask:
        img_full_path = os.sep.join([img_full_path, mask])
    # rblc and rtrc are enforced to be of type list
    if blc is None:
        rblc = len(cshape) * [0]
    else:
        if len(blc) == len(cshape):
            rblc = list(blc[::-1])
            if rblc >= cshape:
                raise RuntimeError(f"blc {blc} >= shape {cshape[::-1]}")
        else:
            raise RuntimeError(
                f"cshape has {len(cshape)} dimensions and blc has {len(blc)}"
            )
    rblc = rblc + [0 for rr in range(5) if rr >= len(mychunks)]
    if trc is None:
        rtrc = cshape
    else:
        if len(trc) == len(cshape):
            rtrc = list(trc[::-1])
            if rtrc > cshape:
                raise RuntimeError(f"trc {trc} >= shape {cshape[::-1]}")
            xblc = rblc if blc is None else list(blc)
            if list(trc) <= xblc:
                raise RuntimeError(f"trc {trc} must be > blc {blc}")
        else:
            raise RuntimeError(
                f"cshape has {len(cshape)} dimensions and blc has {len(blc)}"
            )
    rtrc = rtrc + [1 for rr in range(5) if rr >= len(mychunks)]

    # expand the actual data shape to the full 5 possible dims
    # full_shape = cshape + [1 for rr in range(5) if rr >= len(cshape)]
    # full_shape = list(rtrc - rblc)
    full_chunks = mychunks[::-1] + tuple([1 for rr in range(5) if rr >= len(mychunks)])
    d0slices = []
    for d0 in range(rblc[0], rtrc[0], full_chunks[0]):
        d0len = min(full_chunks[0], rtrc[0] - d0)
        d1slices = []

        for d1 in range(rblc[1], rtrc[1], full_chunks[1]):
            d1len = min(full_chunks[1], rtrc[1] - d1)
            d2slices = []

            for d2 in range(rblc[2], rtrc[2], full_chunks[2]):
                d2len = min(full_chunks[2], rtrc[2] - d2)
                d3slices = []

                for d3 in range(rblc[3], rtrc[3], full_chunks[3]):
                    d3len = min(full_chunks[3], rtrc[3] - d3)
                    d4slices = []

                    for d4 in range(rblc[4], rtrc[4], full_chunks[4]):
                        d4len = min(full_chunks[4], rtrc[4] - d4)

                        shapes = tuple(
                            [d0len, d1len, d2len, d3len, d4len][: len(cshape)]
                        )
                        starts = tuple([d0, d1, d2, d3, d4][: len(cshape)])
                        delayed_array = dask.delayed(_read_image_chunk)(
                            img_full_path, shapes, starts
                        )
                        d4slices += [
                            dask.array.from_delayed(delayed_array, shapes, data_type)
                        ]
                    d3slices += (
                        [dask.array.concatenate(d4slices, axis=4)]
                        if len(cshape) > 4
                        else d4slices
                    )
                d2slices += (
                    [dask.array.concatenate(d3slices, axis=3)]
                    if len(cshape) > 3
                    else d3slices
                )
            d1slices += (
                [dask.array.concatenate(d2slices, axis=2)]
                if len(cshape) > 2
                else d2slices
            )
        d0slices += (
            [dask.array.concatenate(d1slices, axis=1)] if len(cshape) > 1 else d1slices
        )
    ary = dask.array.concatenate(d0slices, axis=0)
    ary = da.expand_dims(ary, new_axes)
    return ary.transpose(transpose_list)


def _read_image_chunk(infile: str, shapes: tuple, starts: tuple) -> np.ndarray:
    with open_table_ro(infile) as tb_tool:
        data: np.ndarray = tb_tool.getcellslice(
            tb_tool.colnames()[0],
            0,
            starts,
            tuple(np.array(starts) + np.array(shapes) - 1),
        )
    return data
