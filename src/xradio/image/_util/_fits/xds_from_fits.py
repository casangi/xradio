import astropy as ap
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from ..common import (
    _compute_linear_world_values,
    _compute_velocity_values,
    _compute_world_sph_dims,
    _convert_beam_to_rad,
    _default_freq_info,
    _doppler_types,
    _freq_from_vel,
    _get_unit,
    _get_xds_dim_order,
    _image_type,
    _l_m_attr_notes,
)
from xradio._utils.coord_math import _deg_to_rad
from xradio._utils.dict_helpers import (
    make_quantity,
    make_frequency_reference_dict,
    make_skycoord_dict,
)
import copy
import dask
import dask.array as da
import numpy as np
import re
from typing import Union
import xarray as xr


def _fits_image_to_xds(
    img_full_path: str, chunks: dict, verbose: bool, do_sky_coords: bool
) -> dict:
    """
    TODO: complete documentation
    Create an xds without any pixel data from metadata from the specified FITS image
    """
    # memmap = True allows only part of data to be loaded into memory
    # may also need to pass mode='denywrite'
    # https://stackoverflow.com/questions/35759713/astropy-io-fits-read-row-from-large-fits-file-with-mutliple-hdus
    hdulist = fits.open(img_full_path, memmap=True)
    attrs, helpers, header = _fits_header_to_xds_attrs(hdulist)
    hdulist.close()
    # avoid keeping reference to mem-mapped fits file
    del hdulist
    xds = _create_coords(helpers, header, do_sky_coords)
    sphr_dims = helpers["sphr_dims"]
    ary = _read_image_array(img_full_path, chunks, helpers, verbose)
    dim_order = _get_xds_dim_order(sphr_dims)
    xds = _add_sky_or_aperture(xds, ary, dim_order, helpers, sphr_dims)
    xds.attrs = attrs
    xds = _add_coord_attrs(xds, helpers)
    if helpers["has_multibeam"]:
        xds = _do_multibeam(xds, img_full_path)
    elif "beam" in helpers and helpers["beam"] is not None:
        xds = _add_beam(xds, helpers)
    return xds


def _add_coord_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    xds = _add_time_attrs(xds, helpers)
    xds = _add_freq_attrs(xds, helpers)
    xds = _add_vel_attrs(xds, helpers)
    xds = _add_l_m_attrs(xds, helpers)
    xds = _add_lin_attrs(xds, helpers)
    return xds


def _add_time_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    time_coord = xds.coords["time"]
    meta = copy.deepcopy(helpers["obsdate"])
    del meta["value"]
    # meta["units"] = [ meta["units"] ]
    # meta['format'] = 'MJD'
    # meta['time_scale'] = meta['refer']
    # del meta['refer']
    time_coord.attrs = meta
    xds.assign_coords(time=time_coord)
    return xds


def _add_freq_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    freq_coord = xds.coords["frequency"]
    meta = {}
    if helpers["has_freq"]:
        meta["rest_frequency"] = make_quantity(helpers["restfreq"], "Hz")
        meta["rest_frequencies"] = [meta["rest_frequency"]]
        # meta["frame"] = helpers["specsys"]
        # meta["units"] = "Hz"
        meta["type"] = "frequency"
        meta["wave_unit"] = "mm"
        freq_axis = helpers["freq_axis"]
        meta["reference_value"] = make_frequency_reference_dict(
            helpers["crval"][freq_axis], ["Hz"], helpers["specsys"]
        )
        # meta["cdelt"] = helpers["cdelt"][freq_axis]
    if not meta:
        # this is the default frequency information CASA creates
        meta = _default_freq_info()
    freq_coord.attrs = copy.deepcopy(meta)
    xds["frequency"] = freq_coord
    return xds


def _add_vel_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    vel_coord = xds.coords["velocity"]
    meta = {"units": ["m/s"]}
    if helpers["has_freq"]:
        meta["doppler_type"] = helpers.get("doppler", "RADIO")
    else:
        meta["doppler_type"] = _doppler_types[0]
    meta["type"] = "doppler"
    vel_coord.attrs = copy.deepcopy(meta)
    xds.coords["velocity"] = vel_coord
    return xds


def _add_l_m_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    attr_note = _l_m_attr_notes()
    for c in ["l", "m"]:
        if c in xds.coords:
            xds[c].attrs = {
                "note": attr_note[c],
            }
    return xds


def _add_lin_attrs(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    """
    if helpers["sphr_dims"]:
        for i, name in zip(helpers["dir_axes"], helpers["sphr_axis_names"]):
            meta = {
                "units": "rad",
                "crval": helpers["crval"][i],
                "cdelt": helpers["cdelt"][i],
            }
            xds.coords[name].attrs = meta
    """
    if not helpers["sphr_dims"]:
        for i, j in zip(helpers["dir_axes"], ("u", "v")):
            meta = {
                "units": "wavelengths",
                "crval": helpers["crval"][i],
                "cdelt": helpers["cdelt"][i],
            }
            xds.coords[j].attrs = meta
    return xds


def _is_freq_like(v: str) -> bool:
    return v.startswith("FREQ") or v == "VOPT" or v == "VRAD"


def _xds_direction_attrs_from_header(helpers: dict, header) -> dict:
    # helpers is modified in place, headers is not modified
    t_axes = helpers["t_axes"]
    p0 = header[f"CTYPE{t_axes[0]}"][-3:]
    p1 = header[f"CTYPE{t_axes[1]}"][-3:]
    if p0 != p1:
        raise RuntimeError(
            f"Projections for direction axes ({p0}, {p1}) differ, but they "
            "must be the same"
        )
    direction = {}
    direction["projection"] = p0
    helpers["projection"] = p0
    ref_sys = header["RADESYS"]
    ref_eqx = header["EQUINOX"]
    if ref_sys == "FK5" and ref_eqx == 2000:
        ref_eqx = "J2000.0"
    helpers["ref_sys"] = ref_sys
    helpers["ref_eqx"] = ref_eqx
    # fits does not support conversion frames
    direction["reference"] = make_skycoord_dict(
        [0.0, 0.0], units=["rad", "rad"], frame=ref_sys
    )
    """
    direction["reference"] = {
        "type": "sky_coord",
        "frame": ref_sys,
        "equinox": ref_eqx,
        "units": ["rad", "rad"],
        "value": [0.0, 0.0],
    }
    """
    dir_axes = helpers["dir_axes"]
    ddata = []
    dunits = []
    for i in dir_axes:
        x = helpers["crval"][i] * u.Unit(_get_unit(helpers["cunit"][i]))
        x = x.to("rad")
        ddata.append(x.value)
        # direction["reference"]["value"][i] = x.value
        x = helpers["cdelt"][i] * u.Unit(_get_unit(helpers["cunit"][i]))
        dunits.append(x.to("rad"))
    direction["reference"] = make_skycoord_dict(ddata, units=dunits, frame=ref_sys)
    direction["reference"]["attrs"]["equinox"] = ref_eqx.lower()
    direction["latpole"] = make_quantity(
        header["LATPOLE"] * _deg_to_rad, "rad", dims=["l", "m"]
    )
    direction["lonpole"] = make_quantity(
        header["LONPOLE"] * _deg_to_rad, "rad", dims=["l", "m"]
    )
    pc = np.zeros([2, 2])
    for i in (0, 1):
        for j in (0, 1):
            # dir_axes are now 0-based, but fits needs 1-based
            try:
                pc[i][j] = header[f"PC{dir_axes[i]+1}_{dir_axes[j]+1}"]
            except KeyError:
                try:
                    pc[i][j] = header[f"PC0{dir_axes[i]+1}_0{dir_axes[j]+1}"]
                except KeyError:
                    raise RuntimeError(
                        f"Could not find PC{dir_axes[i]+1}_{dir_axes[j]+1} or "
                        f"PC0{dir_axes[i]+1}_0{dir_axes[j]+1} in FITS header"
                    )
    direction["pc"] = pc
    # Is there really no fits header parameter for projection_parameters?
    direction["projection_parameters"] = np.array([0.0, 0.0])
    return direction


def _fits_header_c_values_to_metadata(helpers: dict, header) -> None:
    # The helpers dict is modified in place. header is not modified
    ctypes = []
    shape = []
    crval = []
    cdelt = []
    crpix = []
    cunit = []
    for i in range(1, helpers["naxes"] + 1):
        ax_type = header[f"CTYPE{i}"]
        ctypes.append(ax_type)
        shape.append(header[f"NAXIS{i}"])
        crval.append(header[f"CRVAL{i}"])
        cdelt.append(header[f"CDELT{i}"])
        # FITS 1-based to python 0-based
        crpix.append(header[f"CRPIX{i}"] - 1)
        cunit.append(header[f"CUNIT{i}"])
    helpers["ax_type"] = ax_type
    helpers["shape"] = shape
    helpers["ctype"] = ctypes
    helpers["crval"] = crval
    helpers["cdelt"] = cdelt
    helpers["crpix"] = crpix
    helpers["cunit"] = cunit


def _get_telescope_metadata(helpers: dict, header) -> dict:
    # The helpers dict is modified in place. header is not modified
    tel = {}
    tel["name"] = header["TELESCOP"]
    if "OBSGEO-x" in header:
        x = header["OBSGEO-X"]
        y = header["OBSGEO-Y"]
        z = header["OBSGEO-Z"]
        xyz = np.array([x, y, z])
        r = np.sqrt(np.sum(xyz * xyz))
        lat = np.arcsin(z / r)
        long = np.arctan2(y, x)
        tel["position"] = {
            "type": "position",
            # I haven't seen a FITS keyword for reference frame of telescope posiiton
            "ellipsoid": "GRS80",
            "units": ["rad", "rad", "m"],
            "value": np.array([long, lat, r]),
        }
        helpers["tel_pos"] = tel["position"]
    return tel


def _pointing_center_to_metadata(helpers: dict, header) -> dict:
    # Neither helpers or header is modified
    t_axes = helpers["t_axes"]
    long_unit = header[f"CUNIT{t_axes[0]}"]
    lat_unit = header[f"CUNIT{t_axes[1]}"]
    unit = []
    for uu in [long_unit, lat_unit]:
        new_u = u.Unit(_get_unit(uu))
        unit.append(new_u)
    pc_long = float(header[f"CRVAL{t_axes[0]}"]) * unit[0]
    pc_lat = float(header[f"CRVAL{t_axes[1]}"]) * unit[1]
    pc_long = pc_long.to(u.rad).value
    pc_lat = pc_lat.to(u.rad).value
    return {"value": np.array([pc_long, pc_lat]), "initial": True}


def _user_attrs_from_header(header) -> dict:
    # header is not modified
    exclude = [
        "ALTRPIX",
        "ALTRVAL",
        "BITPIX",
        "BSCALE",
        "BTYPE",
        "BUNIT",
        "BZERO",
        "CASAMBM",
        "DATE",
        "DATE-OBS",
        "EQUINOX",
        "EXTEND",
        "HISTORY",
        "LATPOLE",
        "LONPOLE",
        "OBSERVER",
        "ORIGIN",
        "TELESCOP",
        "OBJECT",
        "RADESYS",
        "RESTFRQ",
        "SIMPLE",
        "SPECSYS",
        "TIMESYS",
        "VELREF",
    ]
    regex = r"|".join(
        [
            "^NAXIS\\d?$",
            "^CRVAL\\d$",
            "^CRPIX\\d$",
            "^CTYPE\\d$",
            "^CDELT\\d$",
            "^CUNIT\\d$",
            "^OBSGEO-(X|Y|Z)$",
            "^P(C|V)\\d_\\d$",
        ]
    )
    user = {}
    for k, v in header.items():
        if re.search(regex, k) or k in exclude:
            continue
        user[k.lower()] = v
    return user


def _beam_attr_from_header(helpers: dict, header) -> Union[dict, str, None]:
    # The helpers dict is modified in place. header is not modified
    helpers["has_multibeam"] = False
    if "BMAJ" in header:
        # single global beam
        beam = {
            "bmaj": make_quantity(header["BMAJ"], "deg"),
            "bmin": make_quantity(header["BMIN"], "deg"),
            "pa": make_quantity(header["BPA"], "deg"),
        }
        return _convert_beam_to_rad(beam)
    elif "CASAMBM" in header and header["CASAMBM"]:
        # multi-beam
        helpers["has_multibeam"] = True
        return "mb"
    else:
        # no beam
        return None


def _create_dim_map(helpers: dict, header) -> dict:
    # The helpers dict is modified in place. header is not modified
    t_axes = np.array([0, 0])
    dim_map = {}
    helpers["has_freq"] = False
    # fits indexing starts at 1, not 0
    for i in range(1, helpers["naxes"] + 1):
        ax_type = header[f"CTYPE{i}"]
        if ax_type.startswith("RA-"):
            t_axes[0] = i
        elif ax_type.startswith("DEC-"):
            t_axes[1] = i
        elif ax_type == "STOKES":
            dim_map["polarization"] = i - 1
        elif _is_freq_like(ax_type):
            dim_map["freq"] = i - 1
            helpers["has_freq"] = True
            # helpers["native_type"] = ax_type
        else:
            raise RuntimeError(f"{ax_type} is an unsupported axis")
    helpers["t_axes"] = t_axes
    helpers["dim_map"] = dim_map
    return dim_map


def _fits_header_to_xds_attrs(hdulist: fits.hdu.hdulist.HDUList) -> dict:
    primary = None
    beams = None
    for hdu in hdulist:
        if hdu.name == "PRIMARY":
            primary = hdu
        elif hdu.name == "BEAMS":
            beams = hdu
        else:
            raise RuntimeError(f"Unknown HDU name {hdu.name}")
    if not primary:
        raise RuntimeError(f"No PRIMARY HDU found in fits file")
    header = primary.header
    helpers = {}
    attrs = {}
    naxes = header["NAXIS"]
    helpers["naxes"] = naxes
    dim_map = _create_dim_map(helpers, header)
    _fits_header_c_values_to_metadata(helpers, header)
    if "RESTFRQ" in header:
        helpers["restfreq"] = header["RESTFRQ"]
    if "SPECSYS" in header:
        helpers["specsys"] = header["SPECSYS"]
    t_axes = helpers["t_axes"]
    if (t_axes > 0).all():
        dir_axes = t_axes[:]
        dir_axes = dir_axes - 1
        helpers["dir_axes"] = dir_axes
        dim_map["l"] = dir_axes[0]
        dim_map["m"] = dir_axes[1]
        helpers["dim_map"] = dim_map
    else:
        raise RuntimeError("Could not find both direction axes")
    if dir_axes is not None:
        attrs["direction"] = _xds_direction_attrs_from_header(helpers, header)
    # FIXME read fits data in chunks in case all data too large to hold in memory
    has_mask = da.any(da.isnan(primary.data)).compute()
    attrs["active_mask"] = "MASK0" if has_mask else None
    helpers["has_mask"] = has_mask
    beam = _beam_attr_from_header(helpers, header)
    if beam != "mb":
        helpers["beam"] = beam
    if "BITPIX" in header:
        v = abs(header["BITPIX"])
        if v == 32:
            helpers["dtype"] = "float32"
        elif v == 64:
            helpers["dtype"] = "float64"
        else:
            raise RuntimeError(f'Unhandled data type {header["BITPIX"]}')
    helpers["btype"] = header["BTYPE"] if "BTYPE" in header else None
    helpers["bunit"] = header["BUNIT"] if "BUNIT" in header else None
    attrs["object_name"] = header["OBJECT"] if "OBJECT" in header else None
    obsdate = {}
    obsdate["type"] = "time"
    obsdate["value"] = Time(header["DATE-OBS"], format="isot").mjd
    obsdate["units"] = ["d"]
    obsdate["scale"] = header["TIMESYS"]
    obsdate["format"] = "MJD"
    attrs["obsdate"] = obsdate
    helpers["obsdate"] = obsdate
    attrs["observer"] = header["OBSERVER"]
    attrs["pointing_center"] = _pointing_center_to_metadata(helpers, header)
    attrs["description"] = None
    attrs["telescope"] = _get_telescope_metadata(helpers, header)
    # TODO complete _make_history_xds when spec has been finalized
    # attrs['history'] = _make_history_xds(header)
    attrs["user"] = _user_attrs_from_header(header)
    return attrs, helpers, header


def _make_history_xds(header):
    # TODO complete writing history when we actually have a spec for what
    # the image history is supposed to be, since doing this now may
    # be a waste of time if the final spec turns out to be significantly
    # different from our current ad hoc history xds
    # in astropy, 3506803168 seconds corresponds to 1970-01-01T00:00:00
    history_list = list(header.get("HISTORY"))
    for i in range(len(history_list) - 1, -1, -1):
        if (i == len(history_list) - 1 and history_list[i] == "CASA END LOGTABLE") or (
            i == 0 and history_list[i] == "CASA START LOGTABLE"
        ):
            history_list.pop(i)
        elif history_list[i].startswith(">"):
            # entry continuation line
            history_list[i - 1] = history_list[i - 1] + history_list[i][1:]
            history_list.pop(i)


def _create_coords(
    helpers: dict, header: fits.header, do_sky_coords: bool
) -> xr.Dataset:
    dir_axes = helpers["dir_axes"]
    dim_map = helpers["dim_map"]
    sphr_dims = (
        [dim_map["l"], dim_map["m"]] if ("l" in dim_map) and ("m" in dim_map) else []
    )
    helpers["sphr_dims"] = sphr_dims
    coords = {}
    coords["time"] = _get_time_values(helpers)
    coords["frequency"] = _get_freq_values(helpers)
    coords["polarization"] = _get_pol_values(helpers)
    coords["velocity"] = (["frequency"], _get_velocity_values(helpers))
    if len(sphr_dims) > 0:
        for i, c in enumerate(["l", "m"]):
            idx = sphr_dims[i]
            cdelt_rad = helpers["cdelt"][idx] * u.Unit(_get_unit(helpers["cunit"][idx]))
            cdelt_rad = abs(cdelt_rad.to("rad").value)
            if c == "l":
                # l values increase to the east
                # l follows RA as far as increasing/decreasing, see AIPS Meme 27,
                # change in alpha definition three lines below Figure 2 and the first
                # of the pair of equations 10.
                cdelt_rad = -cdelt_rad
            helpers[c] = {}
            helpers[c]["cunit"] = "rad"
            helpers[c]["cdelt"] = cdelt_rad
            coords[c] = _compute_linear_world_values(
                naxis=helpers["shape"][idx],
                crpix=helpers["crpix"][idx],
                crval=0.0,
                cdelt=cdelt_rad,
            )
        if do_sky_coords:
            pick = lambda mylist: [mylist[i] for i in sphr_dims]
            my_ret = _compute_world_sph_dims(
                projection=helpers["projection"],
                shape=pick(helpers["shape"]),
                ctype=pick(helpers["ctype"]),
                crpix=pick(helpers["crpix"]),
                crval=pick(helpers["crval"]),
                cdelt=pick(helpers["cdelt"]),
                cunit=pick(helpers["cunit"]),
            )
            for j, i in enumerate(dir_axes):
                helpers["cunit"][i] = my_ret["unit"][j]
                helpers["crval"][i] = my_ret["ref_val"][j]
                helpers["cdelt"][i] = my_ret["inc"][j]
            coords[my_ret["axis_name"][0]] = (["l", "m"], my_ret["value"][0])
            coords[my_ret["axis_name"][1]] = (["l", "m"], my_ret["value"][1])
            helpers["sphr_axis_names"] = tuple(my_ret["axis_name"])
    else:
        # Fourier image
        coords["u"], coords["v"] = _get_uv_values(helpers)
    coords["beam_param"] = ["major", "minor", "pa"]
    xds = xr.Dataset(coords=coords)
    return xds


def _get_time_values(helpers):
    return [helpers["obsdate"]["value"]]


def _get_pol_values(helpers):
    # as mapped in casacore Stokes.h
    stokes_map = [
        "Undefined",
        "I",
        "Q",
        "U",
        "V",
        "RR",
        "RL",
        "LR",
        "LL",
        "XX",
        "XY",
        "YX",
        "YY",
        "RX",
        "RY",
        "LX",
        "LY",
        "XR",
        "XL",
        "YR",
        "YL",
        "PP",
        "PQ",
    ]
    idx = helpers["ctype"].index("STOKES")
    if idx >= 0:
        vals = []
        crval = int(helpers["crval"][idx])
        crpix = int(helpers["crpix"][idx])
        cdelt = int(helpers["cdelt"][idx])
        stokes_start_idx = crval - cdelt * crpix
        for i in range(helpers["shape"][idx]):
            stokes_idx = (stokes_start_idx + i) * cdelt
            vals.append(stokes_map[stokes_idx])
        return vals
    else:
        return ["I"]


def _get_freq_values(helpers: dict) -> list:
    vals = []
    ctype = helpers["ctype"]
    if "FREQ" in ctype:
        freq_idx = ctype.index("FREQ")
        helpers["freq_axis"] = freq_idx
        vals = _compute_linear_world_values(
            naxis=helpers["shape"][freq_idx],
            crval=helpers["crval"][freq_idx],
            crpix=helpers["crpix"][freq_idx],
            cdelt=helpers["cdelt"][freq_idx],
        )
        cunit = helpers["cunit"][freq_idx]
        helpers["frequency"] = vals * u.Unit(cunit)
        return vals
    elif "VOPT" in ctype:
        if "restfreq" in helpers:
            restfreq = helpers["restfreq"] * u.Hz
        else:
            raise RuntimeError(
                "Spectral axis in FITS header is velocity, but there is "
                "no rest frequency so converting to frequency is not possible"
            )
        helpers["doppler"] = "Z"
        v_idx = ctype.index("VOPT")
        helpers["freq_idx"] = v_idx
        helpers["freq_axis"] = v_idx
        crval = helpers["crval"][v_idx]
        crpix = helpers["crpix"][v_idx]
        cdelt = helpers["cdelt"][v_idx]
        cunit = helpers["cunit"][v_idx]
        freq, vel = _freq_from_vel(
            crval, cdelt, crpix, cunit, "Z", helpers["shape"][v_idx], restfreq
        )
        helpers["velocity"] = vel["value"] * u.Unit(vel["unit"])
        helpers["crval"][v_idx] = (freq["crval"] * u.Unit(freq["unit"])).to(u.Hz).value
        helpers["cdelt"][v_idx] = (freq["cdelt"] * u.Unit(freq["unit"])).to(u.Hz).value
        return list(freq["value"])
    else:
        return [1420e6]


def _get_velocity_values(helpers: dict) -> list:
    if "velocity" in helpers:
        return helpers["velocity"].to(u.m / u.s).value
    elif "frequency" in helpers:
        v = _compute_velocity_values(
            restfreq=helpers["restfreq"],
            freq_values=helpers["frequency"].to("Hz").value,
            doppler=helpers.get("doppler", "RADIO"),
        )
        helpers["velocity"] = v * (u.m / u.s)
        return v


def _do_multibeam(xds: xr.Dataset, imname: str) -> xr.Dataset:
    """Only run if we are sure there are multiple beams"""
    hdulist = fits.open(imname)
    for hdu in hdulist:
        header = hdu.header
        if "EXTNAME" in header and header["EXTNAME"] == "BEAMS":
            units = (
                u.Unit(header["TUNIT1"]),
                u.Unit(header["TUNIT2"]),
                u.Unit(header["TUNIT3"]),
            )
            nchan = header["NCHAN"]
            npol = header["NPOL"]
            beam_array = np.zeros([1, nchan, npol, 3])
            data = hdu.data
            hdulist.close()
            for t in data:
                beam_array[0, t[3], t[4]] = t[0:3]
            for i in (0, 1, 2):
                beam_array[:, :, :, i] = (
                    (beam_array[:, :, :, i] * units[i]).to("rad").value
                )
            return _create_beam_data_var(xds, beam_array)
    raise RuntimeError(
        "It looks like there should be a BEAMS table but no "
        "such table found in FITS file"
    )


def _add_beam(xds: xr.Dataset, helpers: dict) -> xr.Dataset:
    nchan = xds.sizes["frequency"]
    npol = xds.sizes["polarization"]
    beam_array = np.zeros([1, nchan, npol, 3])
    beam_array[0, :, :, 0] = helpers["beam"]["bmaj"]["data"]
    beam_array[0, :, :, 1] = helpers["beam"]["bmin"]["data"]
    beam_array[0, :, :, 2] = helpers["beam"]["pa"]["data"]
    return _create_beam_data_var(xds, beam_array)


def _create_beam_data_var(xds: xr.Dataset, beam_array: np.array) -> xr.Dataset:
    xdb = xr.DataArray(
        beam_array, dims=["time", "frequency", "polarization", "beam_param"]
    )
    xdb = xdb.rename("BEAM")
    xdb = xdb.assign_coords(beam_param=["major", "minor", "pa"])
    xdb.attrs["units"] = "rad"
    xds["BEAM"] = xdb
    return xds


def _get_uv_values(helpers: dict) -> tuple:
    shape = helpers["shape"]
    ctype = helpers["ctype"]
    unit = helpers["cunit"]
    delt = helpers["cdelt"]
    ref_pix = helpers["crpix"]
    ref_val = helpers["crval"]
    for i, axis in enumerate(["UU", "VV"]):
        idx = ctype.index(axis)
        if idx >= 0:
            z = []
            crpix = ref_pix[i]
            crval = ref_val[i]
            cdelt = delt[i]
            for i in range(shape[idx]):
                f = (i - crpix) * cdelt + crval
                z.append(f)
            if axis == "UU":
                u = z
            else:
                v = z
    return u, v


def _add_sky_or_aperture(
    xds: xr.Dataset,
    ary: Union[np.ndarray, da.array],
    dim_order: list,
    helpers: dict,
    has_sph_dims: bool,
) -> xr.Dataset:
    xda = xr.DataArray(ary, dims=dim_order)
    image_type = helpers["btype"]
    unit = helpers["bunit"]
    xda.attrs[_image_type] = image_type
    xda.attrs["units"] = unit
    name = "SKY" if has_sph_dims else "APERTURE"
    xda = xda.rename(name)
    xds[xda.name] = xda
    if helpers["has_mask"]:
        pp = da if type(xda[0].data) == dask.array.core.Array else np
        mask = pp.isnan(xda)
        mask.attrs = {}
        mask = mask.rename("MASK0")
        xds["MASK0"] = mask
    return xds


def _read_image_array(
    img_full_path: str, chunks: dict, helpers: dict, verbose: bool
) -> da.array:
    # memmap = True allows only part of data to be loaded into memory
    # may also need to pass mode='denywrite'
    # https://stackoverflow.com/questions/35759713/astropy-io-fits-read-row-from-large-fits-file-with-mutliple-hdus
    if isinstance(chunks, dict):
        mychunks = _get_chunk_list(chunks, helpers)
    else:
        raise ValueError(
            f"incorrect type {type(chunks)} for parameter chunks. Must be dict"
        )
    transpose_list, new_axes = _get_transpose_list(helpers)
    data_type = helpers["dtype"]
    rshape = helpers["shape"][::-1]
    full_chunks = mychunks + tuple([1 for rr in range(5) if rr >= len(mychunks)])
    d0slices = []
    blc = tuple(5 * [0])
    trc = tuple(rshape) + tuple([1 for rr in range(5) if rr >= len(mychunks)])
    for d0 in range(blc[0], trc[0], full_chunks[0]):
        d0len = min(full_chunks[0], trc[0] - d0)
        d1slices = []
        for d1 in range(blc[1], trc[1], full_chunks[1]):
            d1len = min(full_chunks[1], trc[1] - d1)
            d2slices = []
            for d2 in range(blc[2], trc[2], full_chunks[2]):
                d2len = min(full_chunks[2], trc[2] - d2)
                d3slices = []
                for d3 in range(blc[3], trc[3], full_chunks[3]):
                    d3len = min(full_chunks[3], trc[3] - d3)
                    d4slices = []
                    for d4 in range(blc[4], trc[4], full_chunks[4]):
                        d4len = min(full_chunks[4], trc[4] - d4)
                        shapes = tuple(
                            [d0len, d1len, d2len, d3len, d4len][: len(rshape)]
                        )
                        starts = tuple([d0, d1, d2, d3, d4][: len(rshape)])
                        delayed_array = dask.delayed(_read_image_chunk)(
                            img_full_path, shapes, starts
                        )
                        d4slices += [da.from_delayed(delayed_array, shapes, data_type)]
                    d3slices += (
                        [da.concatenate(d4slices, axis=4)]
                        if len(rshape) > 4
                        else d4slices
                    )
                d2slices += (
                    [da.concatenate(d3slices, axis=3)] if len(rshape) > 3 else d3slices
                )
            d1slices += (
                [da.concatenate(d2slices, axis=2)] if len(rshape) > 2 else d2slices
            )
        d0slices += [da.concatenate(d1slices, axis=1)] if len(rshape) > 1 else d1slices
    ary = da.concatenate(d0slices, axis=0)
    ary = da.expand_dims(ary, new_axes)
    return ary.transpose(transpose_list)


def _get_chunk_list(chunks: dict, helpers: dict) -> tuple:
    ret_list = list(helpers["shape"])[::-1]
    axis = 0
    ctype = helpers["ctype"]
    for c in ctype[::-1]:
        if c.startswith("RA"):
            if "l" in chunks:
                ret_list[axis] = chunks["l"]
        elif c.startswith("DEC"):
            if "m" in chunks:
                ret_list[axis] = chunks["m"]
        elif c.startswith("FREQ") or c.startswith("VOPT") or c.startswith("VRAD"):
            if "frequency" in chunks:
                ret_list[axis] = chunks["frequency"]
        elif c.startswith("STOKES"):
            if "polarization" in chunks:
                ret_list[axis] = chunks["polarization"]
        else:
            raise RuntimeError(f"Unhandled coordinate type {c}")
        axis += 1
    return tuple(ret_list)


def _get_transpose_list(helpers: dict) -> tuple:
    ctype = helpers["ctype"]
    transpose_list = 5 * [-1]
    # time axis
    transpose_list[0] = 4
    new_axes = [4]
    last_axis = 3
    not_covered = ["l", "m", "u", "v", "s", "f"]
    for i, c in enumerate(ctype[::-1]):
        b = c.lower()
        if b.startswith("ra") or b.startswith("uu"):
            transpose_list[3] = i
            not_covered.remove("l")
            not_covered.remove("u")
        elif b.startswith("dec") or b.startswith("vv"):
            transpose_list[4] = i
            not_covered.remove("m")
            not_covered.remove("v")
        elif (
            b.startswith("frequency")
            or b.startswith("freq")
            or b.startswith("vopt")
            or b.startswith("vrad")
        ):
            transpose_list[1] = i
            not_covered.remove("f")
        elif b.startswith("stok"):
            transpose_list[2] = i
            not_covered.remove("s")
        else:
            raise RuntimeError(f"Unhandled axis name {c}")
    h = {"l": 3, "m": 4, "u": 3, "v": 4, "f": 2, "s": 1}
    for p in not_covered:
        transpose_list[h[p]] = last_axis
        new_axes.append(last_axis)
        last_axis -= 1
    new_axes.sort()
    if transpose_list.count(-1) > 0:
        raise RuntimeError(f"Logic error: axes {axes}, transpose_list {transpose_list}")
    return transpose_list, new_axes


def _read_image_chunk(img_full_path, shapes: tuple, starts: tuple) -> np.ndarray:
    hdulist = fits.open(img_full_path, memmap=True)
    s = []
    for start, length in zip(starts, shapes):
        s.append(slice(start, start + length))
    t = tuple(s)
    z = hdulist[0].data[t]
    hdulist.close()
    # delete to avoid having a reference to a mem-mapped hdulist
    del hdulist
    return z
