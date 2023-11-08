import astropy
from astropy.coordinates import Angle, SkyCoord
from casacore import images, tables
import copy
import dask.array as da
import numpy as np
import os
from typing import Union
import xarray as xr
from .common import __active_mask, __native_types, __object_name, __pointing_center
from ..common import __doppler_types
from ...._utils._casacore.tables import open_table_rw


# TODO move this to a common file to be shared
def __compute_ref_pix(xds: xr.Dataset, direction: dict) -> np.ndarray:
    # TODO more general coordinates
    long = xds.right_ascension
    lat = xds.declination
    ra_crval = long.attrs["wcs"]["crval"]
    dec_crval = lat.attrs["wcs"]["crval"]
    long_close = np.where(np.isclose(long, ra_crval))
    lat_close = np.where(np.isclose(lat, dec_crval))
    if long_close and lat_close:
        long_list = [(i, j) for i, j in zip(long_close[0], long_close[1])]
        lat_list = [(i, j) for i, j in zip(lat_close[0], lat_close[1])]
        common_indices = [t for t in long_list if t in lat_list]
        if len(common_indices) == 1:
            return np.array(common_indices[0])
    cdelt = max(abs(long.attrs["wcs"]["cdelt"]), abs(lat.attrs["wcs"]["cdelt"]))

    # this creates an image of mostly NaNs. The few pixels with values are
    # close to the reference pixel
    ra_diff = long - ra_crval
    dec_diff = lat - dec_crval
    # this returns a 2-tuple of indices where the values in aa are not NaN
    indices_close = np.where(
        ra_diff * ra_diff + dec_diff * dec_diff < 2 * cdelt * cdelt
    )
    # this determines the closest pixel to the reference pixel
    closest = 5e10
    pix = []
    for i, j in zip(indices_close[0], indices_close[1]):
        dra = long[i, j] - ra_crval
        ddec = lat[i, j] - dec_crval
        if dra * dra + ddec * ddec < closest:
            pix = [i, j]
    xds_dir = xds.attrs["direction"]
    # get the actual ref pix
    proj = direction["projection"]
    wcs_dict = {}
    wcs_dict[f"CTYPE1"] = f"RA---{proj}"
    wcs_dict[f"NAXIS1"] = long.shape[0]
    wcs_dict[f"CUNIT1"] = long.attrs["unit"]
    # FITS arrays are 1-based
    wcs_dict[f"CRPIX1"] = pix[0] + 1
    wcs_dict[f"CRVAL1"] = long[pix[0], pix[1]].item(0)
    wcs_dict[f"CDELT1"] = long.attrs["wcs"]["cdelt"]
    wcs_dict[f"CTYPE2"] = f"DEC--{proj}"
    wcs_dict[f"NAXIS2"] = lat.shape[1]
    wcs_dict[f"CUNIT2"] = lat.attrs["unit"]
    # FITS arrays are 1-based
    wcs_dict[f"CRPIX2"] = pix[1] + 1
    wcs_dict[f"CRVAL2"] = lat[pix[0], pix[1]].item(0)
    wcs_dict[f"CDELT2"] = lat.attrs["wcs"]["cdelt"]
    w = astropy.wcs.WCS(wcs_dict)
    x, y = np.indices(w.pixel_shape)
    sky = SkyCoord(
        ra_crval,
        dec_crval,
        frame=xds.attrs["direction"]["system"].lower(),
        equinox=xds.attrs["direction"]["equinox"],
        unit=long.attrs["unit"],
    )
    return w.world_to_pixel(sky)


def __compute_direction_dict(xds: xr.Dataset) -> dict:
    """
    Given xds metadata, compute the direction dict that is valid
    for a CASA image coordinate system
    """
    direction = {}
    xds_dir = xds.attrs["direction"]
    direction["system"] = xds_dir["equinox"]
    direction["projection"] = xds_dir["projection"]
    direction["projection_parameters"] = xds_dir["projection_parameters"]
    long = xds.right_ascension
    lat = xds.declination
    direction["units"] = np.array([long.attrs["unit"], lat.attrs["unit"]], dtype="<U16")
    direction["crval"] = np.array(
        [long.attrs["wcs"]["crval"], lat.attrs["wcs"]["crval"]]
    )
    direction["cdelt"] = np.array(
        [long.attrs["wcs"]["cdelt"], lat.attrs["wcs"]["cdelt"]]
    )
    """
    # get the actual ref pix
    proj = direction['projection']
    wcs_dict = {}
    wcs_dict[f'CTYPE1'] = f'RA---{proj}'
    wcs_dict[f'NAXIS1'] = long.shape[0]
    wcs_dict[f'CUNIT1'] = long.attrs['unit']
    # FITS arrays are 1-based
    wcs_dict[f'CRPIX1'] = 1
    wcs_dict[f'CRVAL1'] = long[0][0].item(0)
    wcs_dict[f'CDELT1'] = long.attrs['wcs']['cdelt']
    wcs_dict[f'CTYPE2'] = f'DEC--{proj}'
    wcs_dict[f'NAXIS2'] = lat.shape[1]
    wcs_dict[f'CUNIT2'] = lat.attrs['unit']
    # FITS arrays are 1-based
    wcs_dict[f'CRPIX2'] = 1
    wcs_dict[f'CRVAL2'] = lat[0][0].item(0)
    wcs_dict[f'CDELT2'] = lat.attrs['wcs']['cdelt']
    print('*** wcs_dict', wcs_dict)
    w = astropy.wcs.WCS(wcs_dict)
    x, y = np.indices(w.pixel_shape)
    sky = SkyCoord(direction[
        'crval'][0], direction['crval'][1],
        frame=xds.attrs['direction']['system'].lower(),
        equinox=xds.attrs['direction']['equinox'],
        unit=long.attrs['unit']
    )
    crpix = w.world_to_pixel(sky)
    """
    crpix = __compute_ref_pix(xds, direction)
    direction["crpix"] = np.array([crpix[0], crpix[1]])
    direction["pc"] = xds_dir["pc"]
    direction["axes"] = ["Right Ascension", "Declination"]
    direction["conversionSystem"] = direction["system"]
    for s in ["longpole", "latpole"]:
        direction[s] = Angle(str(xds_dir[s]["value"]) + xds_dir[s]["unit"]).deg
    return direction


def __compute_spectral_dict(
    xds: xr.Dataset,
    direction: dict,
    obsdate: dict,
    tel_pos: dict,
) -> dict:
    """
    Given xds metadata, compute the spectral dict that is valid
    for a CASA image coordinate system
    """
    spec = {}
    spec_conv = copy.deepcopy(xds.frequency.attrs["conversion"])
    for k in ("direction", "epoch", "position"):
        spec_conv[k]["type"] = k
    spec_conv["direction"]["refer"] = spec_conv["direction"]["system"]
    del spec_conv["direction"]["system"]
    if (
        spec_conv["direction"]["refer"] == "FK5"
        and spec_conv["direction"]["equinox"] == "J2000"
    ):
        spec_conv["direction"]["refer"] = "J2000"
    del spec_conv["direction"]["equinox"]
    spec["conversion"] = spec_conv
    spec["formatUnit"] = ""
    spec["name"] = "Frequency"
    spec["nativeType"] = __native_types.index(xds.frequency.attrs["native_type"])
    spec["restfreq"] = xds.frequency.attrs["restfreq"]
    spec["restfreqs"] = copy.deepcopy(xds.frequency.attrs["restfreqs"])
    spec["system"] = xds.frequency.attrs["system"]
    spec["unit"] = xds.frequency.attrs["unit"]
    spec["velType"] = __doppler_types.index(xds.velocity.attrs["doppler_type"])
    spec["velUnit"] = xds.velocity.attrs["unit"]
    spec["version"] = 2
    spec["waveUnit"] = xds.frequency.attrs["wave_unit"]
    spec_wcs = copy.deepcopy(xds.frequency.attrs["wcs"])
    spec_wcs["ctype"] = "FREQ"
    spec_wcs["pc"] = 1.0
    spec_wcs["crpix"] = (spec_wcs["crval"] - xds.frequency.values[0]) / spec_wcs[
        "cdelt"
    ]
    spec["wcs"] = spec_wcs
    return spec


def __coord_dict_from_xds(xds: xr.Dataset) -> dict:
    coord = {}
    coord["telescope"] = xds.attrs["telescope"]["name"]
    coord["observer"] = xds.attrs["observer"]
    obsdate = {}
    obsdate["refer"] = xds.coords["time"].attrs["time_scale"]
    obsdate["type"] = "epoch"
    obsdate["m0"] = {}
    obsdate["m0"]["unit"] = xds.coords["time"].attrs["unit"]
    obsdate["m0"]["value"] = xds.coords["time"].values[0]
    # obsdate['format'] = xds.time.attrs['format']
    coord["obsdate"] = obsdate
    coord["pointingcenter"] = xds.attrs[__pointing_center].copy()
    coord["telescopeposition"] = xds.attrs["telescope"]["position"].copy()
    coord["direction0"] = __compute_direction_dict(xds)
    coord["stokes1"] = {
        "axes": np.array(["Stokes"], dtype="<U16"),
        "cdelt": np.array([1.0]),
        "crpix": np.array([0.0]),
        "crval": np.array([1.0]),
        "pc": np.array([[1.0]]),
        "stokes": np.array(xds.polarization.values, dtype="<U16"),
    }
    coord["spectral2"] = __compute_spectral_dict(
        xds, coord["direction0"], coord["obsdate"], coord["telescopeposition"]
    )
    coord["pixelmap0"] = np.array([0, 1])
    coord["pixelmap1"] = np.array([2])
    coord["pixelmap2"] = np.array([3])
    coord["pixelreplace0"] = np.array([0.0, 0.0])
    coord["pixelreplace1"] = np.array([0.0])
    coord["pixelreplace2"] = np.array([0.0])
    coord["worldmap0"] = np.array([0, 1], dtype=np.int32)
    coord["worldmap1"] = np.array([2], dtype=np.int32)
    coord["worldmap2"] = np.array([3], dtype=np.int32)
    # coord['worldreplace0'] = coord['direction0']['crval']
    # this probbably needs some verification
    coord["worldreplace0"] = [0.0, 0.0]
    coord["worldreplace1"] = np.array(coord["stokes1"]["crval"])
    coord["worldreplace2"] = np.array([xds.frequency.attrs["wcs"]["crval"]])
    return coord


def __history_from_xds(xds: xr.Dataset, image: str) -> None:
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


def __imageinfo_dict_from_xds(xds: xr.Dataset) -> dict:
    ii = {}
    ii["image_type"] = (
        xds.sky.attrs["image_type"] if "image_type" in xds.sky.attrs else ""
    )
    ii["objectname"] = xds.attrs[__object_name]
    if "beam" in xds.data_vars:
        # multi beam
        pp = {}
        pp["nChannels"] = len(xds.frequency)
        pp["nStokes"] = len(xds.polarization)
        bu = xds.beam.attrs["unit"]
        chan = 0
        polarization = 0
        bv = xds.beam.values
        for i in range(pp["nChannels"] * pp["nStokes"]):
            bp = bv[0][polarization][chan][:]
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
    elif "beam" in xds.attrs and xds.attrs["beam"]:
        # do nothing if xds.attrs['beam'] is None
        ii["restoringbeam"] = xds.attrs["beam"]
    return ii


def __write_casa_data(xds: xr.Dataset, image_full_path: str) -> None:
    sky_ap = "sky" if "sky" in xds else "apeature"
    if xds[sky_ap].shape[0] != 1:
        raise Exception("XDS can only be converted if it has exactly one time plane")
    casa_image_shape = (
        xds[sky_ap]
        .isel(time=0)
        .transpose(*("frequency", "polarization", "m", "l"))
        .shape[::-1]
    )
    active_mask = xds.attrs["active_mask"] if __active_mask in xds.attrs else ""
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
    __write_initial_image(xds, image_full_path, active_mask, casa_image_shape[::-1])
    for v in myvars:
        __write_pixels(v, active_mask, image_full_path, xds)
    for name, v in arr_masks.items():
        __write_pixels(name, active_mask, image_full_path, xds, v)
    if masks:
        with open_table_rw(image_full_path) as tb:
            tb.putkeyword("masks", masks_rec)
            tb.putkeyword("Image_defaultmask", active_mask)


def __write_initial_image(
    xds: xr.Dataset, imagename: str, maskname: str, image_shape: tuple
):
    image_full_path = os.path.expanduser(imagename)
    # create the image and then delete the object
    if not maskname:
        maskname = ""
    casa_image = images.image(image_full_path, maskname=maskname, shape=image_shape)
    del casa_image


def __write_image_block(xda: xr.DataArray, outfile: str, blc: tuple) -> None:
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


def __write_pixels(
    v: str,
    active_mask: str,
    image_full_path: str,
    xds: xr.Dataset,
    value: xr.DataArray = None,
) -> None:
    flip = False
    if v == "sky" or v == "apeture":
        filename = image_full_path
    else:
        # mask
        flip = True
        filename = os.sep.join([image_full_path, v])
        # the default mask has already been written in __xds_to_casa_image()
        if not os.path.exists(filename):
            tb = tables.table(os.sep.join([image_full_path, active_mask]))
            tb.copy(filename, deep=True, valuecopy=True)
            tb.close()
    arr = xds[v] if v in xds.data_vars else value
    arr = arr.isel(time=0).transpose(*("frequency", "polarization", "m", "l"))
    chunk_bounds = arr.chunks
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
                    __write_image_block(sub_arr, filename, blc)
                    loc3 += i3
                loc2 += i2
            loc1 += i1
        loc0 += i0
