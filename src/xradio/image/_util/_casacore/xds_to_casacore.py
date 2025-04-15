import copy
import os

import dask.array as da
import numpy as np
import xarray as xr
from astropy.coordinates import Angle
from casacore import tables

from .common import _active_mask, _create_new_image, _object_name, _pointing_center
from ..common import _aperture_or_sky, _compute_sky_reference_pixel, _doppler_types
from ...._utils._casacore.tables import open_table_rw


def _compute_direction_dict(xds: xr.Dataset) -> dict:
    """
    Given xds metadata, compute the direction dict that is valid
    for a CASA image coordinate system
    """
    direction = {}
    xds_dir = xds.attrs["direction"]
    direction["system"] = xds_dir["reference"]["attrs"]["equinox"].upper()
    if direction["system"] == "J2000.0":
        direction["system"] = "J2000"
    direction["projection"] = xds_dir["projection"]
    direction["projection_parameters"] = xds_dir["projection_parameters"]
    direction["units"] = np.array(xds_dir["reference"]["attrs"]["units"], dtype="<U16")
    direction["crval"] = np.array(xds_dir["reference"]["data"])
    direction["cdelt"] = np.array((xds.l[1] - xds.l[0], xds.m[1] - xds.m[0]))
    direction["crpix"] = _compute_sky_reference_pixel(xds)
    direction["pc"] = np.array(xds_dir["pc"])
    direction["axes"] = ["Right Ascension", "Declination"]
    direction["conversionSystem"] = direction["system"]
    for s in ["longpole", "latpole"]:
        m = "lonpole" if s == "longpole" else s
        # lonpole, latpole are numerical values in degrees in casa images
        direction[s] = Angle(
            str(xds_dir[m]["data"]) + xds_dir[m]["attrs"]["units"][0]
        ).deg
    return direction


def _compute_linear_dict(xds: xr.Dataset) -> dict:
    linear = {}
    u = xds["u"].attrs
    v = xds["v"].attrs
    linear["crval"] = np.array([u["crval"], v["crval"]])
    linear["cdelt"] = np.array([u["cdelt"], v["cdelt"]])
    linear["axes"] = ["UU", "VV"]
    linear["units"] = [u["units"], v["units"]]
    lu = len(xds.coords["u"])
    lv = len(xds.coords["v"])
    linear["crpix"] = np.array([0.0, 0.0], dtype=np.float64)
    # np.interp() appears to require the x coordinate be monotonically increasing
    for i, w, z in zip([0, 1], [u, v], ["u", "v"]):
        x = xds.coords[z].values
        y = np.array(range(len(x)), dtype=np.float64)
        if w["cdelt"] < 0:
            x = x[::-1]
            y = y[::-1]
        linear["crpix"][i] = np.interp(0, x, y)
    linear["pc"] = np.array([[1.0, 0.0], [0.0, 1.0]])
    return linear


def _compute_spectral_dict(xds: xr.Dataset) -> dict:
    """
    Given xds metadata, compute the spectral dict that is valid
    for a CASA image coordinate system
    """
    spec = {}
    spec["formatUnit"] = ""
    spec["name"] = "Frequency"
    # spec["nativeType"] = _native_types.index(xds.frequency.attrs["native_type"])
    # FREQ
    spec["nativeType"] = 0
    spec["restfreq"] = xds.frequency.attrs["rest_frequency"]["data"]
    # spec["restfreqs"] = copy.deepcopy(xds.frequency.attrs["restfreqs"]["value"])
    spec["restfreqs"] = [spec["restfreq"]]
    spec["system"] = xds.frequency.attrs["reference_value"]["attrs"]["observer"].upper()
    u = xds.frequency.attrs["reference_value"]["attrs"]["units"]
    spec["unit"] = u if isinstance(u, str) else u[0]
    spec["velType"] = _doppler_types.index(xds.velocity.attrs["doppler_type"])
    u = xds.velocity.attrs["units"]
    spec["velUnit"] = u if isinstance(u, str) else u[0]
    spec["version"] = 2
    spec["waveUnit"] = xds.frequency.attrs["wave_unit"]
    wcs = {}
    wcs["ctype"] = "FREQ"
    wcs["pc"] = 1.0
    wcs["crval"] = xds.frequency.attrs["reference_value"]["data"]
    wcs["cdelt"] = xds.frequency.values[1] - xds.frequency.values[0]
    wcs["crpix"] = (wcs["crval"] - xds.frequency.values[0]) / wcs["cdelt"]
    spec["wcs"] = wcs
    return spec


def _coord_dict_from_xds(xds: xr.Dataset) -> dict:
    coord = {}
    coord["telescope"] = xds.attrs["telescope"]["name"]
    coord["observer"] = xds.attrs["observer"]
    obsdate = {}
    obsdate["refer"] = xds.coords["time"].attrs["scale"]
    obsdate["type"] = "epoch"
    obsdate["m0"] = {}
    obsdate["m0"]["unit"] = xds.coords["time"].attrs["units"][0]
    obsdate["m0"]["value"] = xds.coords["time"].values[0]
    coord["obsdate"] = obsdate
    coord["pointingcenter"] = xds.attrs[_pointing_center].copy()
    if "position" in xds.attrs["telescope"]:
        telpos = {}
        telpos["refer"] = xds.attrs["telescope"]["position"]["ellipsoid"]
        if xds.attrs["telescope"]["position"]["ellipsoid"] == "GRS80":
            telpos["refer"] = "ITRF"
        for i in range(3):
            telpos[f"m{i}"] = {
                "unit": xds.attrs["telescope"]["position"]["units"][i],
                "value": xds.attrs["telescope"]["position"]["value"][i],
            }
        telpos["type"] = "position"
        coord["telescopeposition"] = telpos
    if "l" in xds.coords:
        coord["direction0"] = _compute_direction_dict(xds)
    else:
        coord["linear0"] = _compute_linear_dict(xds)
    coord["stokes1"] = {
        "axes": np.array(["Stokes"], dtype="<U16"),
        "cdelt": np.array([1.0]),
        "crpix": np.array([0.0]),
        "crval": np.array([1.0]),
        "pc": np.array([[1.0]]),
        "stokes": np.array(xds.polarization.values, dtype="<U16"),
    }
    coord["spectral2"] = _compute_spectral_dict(xds)
    coord["pixelmap0"] = np.array([0, 1])
    coord["pixelmap1"] = np.array([2])
    coord["pixelmap2"] = np.array([3])
    coord["pixelreplace0"] = np.array([0.0, 0.0])
    coord["pixelreplace1"] = np.array([0.0])
    coord["pixelreplace2"] = np.array([0.0])
    coord["worldmap0"] = np.array([0, 1], dtype=np.int32)
    coord["worldmap1"] = np.array([2], dtype=np.int32)
    coord["worldmap2"] = np.array([3], dtype=np.int32)
    # this probbably needs some verification
    coord["worldreplace0"] = [0.0, 0.0]
    coord["worldreplace1"] = np.array(coord["stokes1"]["crval"])
    # print("spectral", coord["spectral2"])
    coord["worldreplace2"] = np.array(coord["spectral2"]["wcs"]["crval"])
    return coord


def _history_from_xds(xds: xr.Dataset, image: str) -> None:
    nrows = len(xds.history.row) if "row" in xds.data_vars else 0
    if nrows > 0:
        # TODO need to implement nrows == 0 case
        with open_table_rw(os.sep.join([image, "logtable"])) as tb:
            tb.addrows(nrows + 1)
            for c in ["TIME", "PRIORITY", "MESSAGE", "LOCATION", "OBJECT_ID"]:
                vals = xds.history[c].values
                if c == "TIME":
                    k = time.time() + 40587 * 86400
                elif c == "PRIORITY":
                    k = "INFO"
                elif c == "MESSAGE":
                    k = (
                        "Wrote xds to "
                        + os.path.basename(image)
                        + " using cngi_io.xds_to_casa_image_2()"
                    )
                elif c == "LOCATION":
                    k = "cngi_io.xds_to_casa_image_2"
                elif c == "OBJECT_ID":
                    k = ""
                vals = np.append(vals, k)
                tb.putcol(c, vals)


def _imageinfo_dict_from_xds(xds: xr.Dataset) -> dict:
    ii = {}
    ap_sky = _aperture_or_sky(xds)
    ii["image_type"] = (
        xds[ap_sky].attrs["image_type"] if "image_type" in xds[ap_sky].attrs else ""
    )
    ii["objectname"] = xds.attrs[_object_name]
    if "BEAM" in xds.data_vars:
        # multi beam
        pp = {}
        pp["nChannels"] = xds.sizes["frequency"]
        pp["nStokes"] = xds.sizes["polarization"]
        bu = xds.BEAM.attrs["units"]
        chan = 0
        polarization = 0
        bv = xds.BEAM.values
        for i in range(pp["nChannels"] * pp["nStokes"]):
            bp = bv[0][chan][polarization][:]
            b = {
                "major": {"unit": bu, "value": bp[0]},
                "minor": {"unit": bu, "value": bp[1]},
                "positionangle": {"unit": bu, "value": bp[2]},
            }
            pp["*" + str(pp["nChannels"] * polarization + chan)] = b
            chan += 1
            if chan >= pp["nChannels"]:
                chan = 0
                polarization += 1
        ii["perplanebeams"] = pp
    """
    elif "beam" in xds.attrs and xds.attrs["beam"]:
        # do nothing if xds.attrs['beam'] is None
        ii["restoringbeam"] = copy.deepcopy(xds.attrs["beam"])
        for k in ["major", "minor", "pa"]:
            # print("*** ", k, ii["restoringbeam"][k])
            del ii["restoringbeam"][k]["dims"]
            ii["restoringbeam"][k]["unit"] = ii["restoringbeam"][k]["attrs"]["units"][0]
            del ii["restoringbeam"][k]["attrs"]
            ii["restoringbeam"][k]["value"] = ii["restoringbeam"][k]["data"]
            del ii["restoringbeam"][k]["data"]
        ii["restoringbeam"]["positionangle"] = copy.deepcopy(ii["restoringbeam"]["pa"])
        del ii["restoringbeam"]["pa"]
    """
    return ii


def _write_casa_data(xds: xr.Dataset, image_full_path: str) -> None:
    sky_ap = _aperture_or_sky(xds)
    if xds[sky_ap].shape[0] != 1:
        raise RuntimeError("XDS can only be converted if it has exactly one time plane")
    trans_coords = (
        ("frequency", "polarization", "m", "l")
        if sky_ap == "SKY"
        else ("frequency", "polarization", "v", "u")
    )
    casa_image_shape = xds[sky_ap].isel(time=0).transpose(*trans_coords).shape[::-1]
    active_mask = xds.attrs["active_mask"] if _active_mask in xds.attrs else ""
    masks = []
    masks_rec = {}
    mask_rec = {
        "box": {
            "blc": np.array([1.0, 1.0, 1.0, 1.0]),
            "comment": "",
            "isRegion": 1,
            "name": "LCBox",
            "oneRel": True,
            "shape": np.array(casa_image_shape),
            "trc": np.array(casa_image_shape),
        },
        "comment": "",
        "isRegion": 1,
        "name": "LCPagedMask",
    }
    for m in xds.data_vars:
        attrs = xds[m].attrs
        if "image_type" in attrs and attrs["image_type"] == "Mask":
            masks_rec[m] = mask_rec
            masks_rec[m]["mask"] = f"Table: {os.sep.join([image_full_path, m])}"
            masks.append(m)
    myvars = [sky_ap]
    myvars.extend(masks)
    # xr.apply_ufunc seems to like stripping attributes from xds and its coordinates,
    # so make a deep copy of the pertinent thing to be handled in xr.apply_ufunc()
    nan_mask = xr.apply_ufunc(da.isnan, xds[sky_ap].copy(deep=True), dask="allowed")
    # test if nan pixels are all already masked in active_mask.
    # if so, don't worry about masking them again, just use active_mask
    # which already masks all nans
    arr_masks = {}
    do_mask_nans = False
    there_are_nans = nan_mask.any()
    if there_are_nans:
        has_active_mask = bool(active_mask)
        if not has_active_mask:
            do_mask_nans = True
        else:
            notted_active_mask = xr.apply_ufunc(
                da.logical_not, xds[active_mask].copy(deep=True), dask="allowed"
            )
            some_nans_are_not_already_masked = xr.apply_ufunc(
                da.logical_and, nan_mask, notted_active_mask, dask="allowed"
            )
            do_mask_nans = some_nans_are_not_already_masked.any()
    if do_mask_nans:
        mask_name = "mask_xds_nans"
        i = 0
        while mask_name in masks:
            mask_name = f"mask_xds_nans{i}"
            i += 1
        masks_rec[mask_name] = mask_rec
        masks_rec[mask_name][
            "mask"
        ] = f"Table: {os.sep.join([image_full_path, mask_name])}"
        masks.append(mask_name)
        arr_masks[mask_name] = nan_mask
        if active_mask:
            mask_name = f"mask_xds_nans_or_{active_mask}"
            i = 0
            while mask_name in masks:
                mask_name = f"mask_xds_nans{i}"
                i += 1
            masks_rec[mask_name] = mask_rec
            masks_rec[mask_name][
                "mask"
            ] = f"Table: {os.sep.join([image_full_path, mask_name])}"
            masks.append(mask_name)
            # This command strips attributes at various places for no
            # apparent reason, so make a copy of the xds to run it on
            arr_masks[mask_name] = xr.apply_ufunc(
                da.logical_or,
                xds[active_mask].copy(deep=True),
                nan_mask,
                dask="allowed",
            )
        active_mask = mask_name
    data_type = "complex" if "u" in xds.coords else "float"
    _write_initial_image(xds, image_full_path, active_mask, casa_image_shape[::-1])
    for v in myvars:
        _write_pixels(v, active_mask, image_full_path, xds)
    for name, v in arr_masks.items():
        _write_pixels(name, active_mask, image_full_path, xds, v)
    if masks:
        with open_table_rw(image_full_path) as tb:
            tb.putkeyword("masks", masks_rec)
            tb.putkeyword("Image_defaultmask", active_mask)


def _write_initial_image(
    xds: xr.Dataset, imagename: str, maskname: str, image_shape: tuple
) -> None:
    if not maskname:
        maskname = ""
    for dv in ["SKY", "APERTURE"]:
        if dv in xds.data_vars:
            value = xds[dv][0, 0, 0, 0, 0].values.item()
            if xds[dv][0, 0, 0, 0, 0].values.dtype == "float32":
                value = "default"
            break
    image_full_path = os.path.expanduser(imagename)
    with _create_new_image(
        image_full_path, mask=maskname, shape=image_shape, value=value
    ) as casa_image:
        # just create the image, don't do anythong with it
        pass


def _write_image_block(xda: xr.DataArray, outfile: str, blc: tuple) -> None:
    """
    Write image xda chunk to the corresponding image table slice
    """
    # trigger the DAG for this chunk and return values while the table is
    # unlocked
    values = xda.compute().values
    with open_table_rw(outfile) as tb_tool:
        tb_tool.putcellslice(
            tb_tool.colnames()[0],
            0,
            values,
            blc,
            tuple(np.array(blc) + np.array(values.shape) - 1),
        )


def _write_pixels(
    v: str,
    active_mask: str,
    image_full_path: str,
    xds: xr.Dataset,
    value: xr.DataArray = None,
) -> None:
    flip = False
    if v == "SKY" or v == "APERTURE":
        filename = image_full_path
    else:
        # mask
        flip = True
        filename = os.sep.join([image_full_path, v])
        # the default mask has already been written in _xds_to_casa_image()
        if not os.path.exists(filename):
            tb = tables.table(os.sep.join([image_full_path, active_mask]))
            tb.copy(filename, deep=True, valuecopy=True)
            tb.close()
    if "l" in xds.coords:
        trans_coords = ("frequency", "polarization", "m", "l")
    elif "u" in xds.coords:
        trans_coords = ("frequency", "polarization", "v", "u")
    else:
        raise RuntimeError(f"Unhandled coords combination {xds.coords.keys()}")
    trans_coord = ("frequency", "polarization", "m", "l")
    arr = xds[v] if v in xds.data_vars else value
    arr = arr.isel(time=0).transpose(*trans_coords)
    chunk_bounds = tuple(zip(arr.shape)) if arr.chunks is None else arr.chunks
    b = [0, 0, 0, 0]
    loc0, loc1, loc2, loc3 = (0, 0, 0, 0)
    for i0 in chunk_bounds[0]:
        b[0] = loc0
        s0 = slice(b[0], b[0] + i0)
        loc1 = 0
        for i1 in chunk_bounds[1]:
            b[1] = loc1
            s1 = slice(b[1], b[1] + i1)
            loc2 = 0
            for i2 in chunk_bounds[2]:
                b[2] = loc2
                s2 = slice(b[2], b[2] + i2)
                loc3 = 0
                for i3 in chunk_bounds[3]:
                    b[3] = loc3
                    blc = tuple(b)
                    s3 = slice(b[3], b[3] + i3)
                    sub_arr = arr[s0, s1, s2, s3]
                    if flip:
                        sub_arr = np.logical_not(sub_arr)
                    _write_image_block(sub_arr, filename, blc)
                    loc3 += i3
                loc2 += i2
            loc1 += i1
        loc0 += i0
