"""
Upgrade of image datasets written by xradio 1.2.3 and earlier.

Image zarr stores written by xradio 1.2.3 and earlier predate the image schema
(:py:class:`xradio.image.schema.ImageXds`): stores made from
``make_empty_sky_image`` and its siblings (for example the outputs of
AstroVIPER) have the dataset type ``"image"``, and stores converted from CASA
or FITS images lack the frequency ``units`` and ``frame``.
:func:`upgrade_legacy_image_attrs` brings their attributes up to the current
conventions, so that they open like images written today. ``open_image`` and
``load_image`` apply it to every zarr store they read; writing the opened
dataset back to zarr stores the upgraded attributes (for example to re-bless
test truth stores).
"""

from __future__ import annotations

import copy

import numpy as np
import xarray as xr

from xradio._utils.logging import xradio_logger
from xradio.image._util.common import _hz_per_unit, _l_m_attr_notes
from xradio.image._util.conventions import (
    CASACORE_EPOCH_REF_TO_SCALE,
    normalize_spectral_frame,
    spectral_frame_to_observer,
)

#: astropy time scales (astropy.time.Time.SCALES)
_ASTROPY_TIME_SCALES = ("tai", "tcb", "tcg", "tdb", "tt", "ut1", "utc", "local")
#: The ``note`` attributes of the l and m coordinates written by xradio <= 1.2.3,
#: which described l and m as angles (they are projection plane coordinates).
_LEGACY_L_M_NOTES = {
    "l": "l is the angle measured from the reference direction to the east. "
    "So l = x*cdelt, where x is the number of pixels from the reference direction. "
    "See AIPS Memo #27, Section III.",
    "m": "m is the angle measured from the reference direction to the north. "
    "So m = y*cdelt, where y is the number of pixels from the reference direction. "
    "See AIPS Memo #27, Section III.",
}
#: Attributes of the frequency coordinate that hold frequency quantities.
_FREQUENCY_QUANTITY_ATTRS = (
    "reference_frequency",
    "rest_frequency",
    "rest_frequencies",
    "channel_width",
)
#: Data group roles that ``open_image`` of xradio <= 1.2.3 gave the
#: deconvolution products of a list of tclean images (variables ``MODEL``,
#: ``RESIDUAL`` and ``DIRTY`` in the group of the sky image). They are now the
#: sky images of data groups of their own.
_LEGACY_SKY_ROLES = ("model", "residual", "dirty")
#: Data group role of xradio <= 1.2.3 for the deconvolution mask (variable
#: ``MASK_DECONVOLVE``), now the role ``mask``.
_LEGACY_MASK_ROLE = "mask_deconvolve"
#: Roles of a data group that belong to its sky image, which a data group
#: made for a legacy sky role does not share.
_SKY_IMAGE_ROLES = ("sky", "flag", "beam_fit_params_sky")
#: Data group entries that hold text, not the name of a data variable.
_TEXT_ROLES = ("description", "date")


def upgrade_legacy_image_attrs(xds: xr.Dataset) -> xr.Dataset:
    """
    Bring the attributes of an image dataset written by xradio <= 1.2.3 up to
    the current image schema conventions.

    The upgrade only fills in or translates what the current readers and
    factories write, so datasets that already follow the conventions are
    returned unchanged and the upgrade can be applied to any image dataset
    read from zarr:

    * the dataset ``type`` ``"image"`` becomes ``"image_dataset"``;
    * the frequency coordinate gets ``units`` ``"Hz"`` (values and frequency
      attributes in another frequency unit are converted to Hz), the casacore
      ``frame`` of its reference frequency observer when ``frame`` is missing,
      and observers in the schema vocabulary (for example ``"geo"`` becomes
      ``"gcrs"`` and ``"bary"`` or ``"barycent"`` becomes ``"BARY"``, see
      :mod:`xradio.image._util.conventions`);
    * the time coordinate gets ``type`` ``"time"`` and the format ``"mjd"``
      when it has none (casacore epochs count from MJD 0); a scale without an
      astropy equivalent (a sidereal or unset casacore epoch) becomes
      ``"utc"``, with a warning, and the ``obsdate`` attributes with such a
      scale are dropped, as the CASA image reader does;
    * ``u`` and ``v`` coordinates described by a quantity dictionary (as
      ``make_empty_aperture_image`` and ``make_empty_lmuv_image`` wrote them)
      get the ``units``, ``crval`` and ``cdelt`` attributes the CASA image
      reader writes;
    * ``u`` and ``v`` coordinates without ``units`` (as
      ``ImageXds.add_uv_coordinates`` added them, in wavelengths) get the
      ``units`` ``"lambda"``;
    * the ``note`` of the ``l`` and ``m`` coordinates, which described them as
      angles, is replaced by the current one (projection plane coordinates);
    * the fk4 reference direction of a ``make_empty_*`` image (dataset type
      ``"image"``), which those factories gave the equinox ``"j2000.0"``, gets
      the FK4 equinox ``"b1950.0"``;
    * the data group roles of the deconvolution products of a list of tclean
      images (``model``, ``residual``, ``dirty`` and ``mask_deconvolve``)
      become data groups of their own and the role ``mask`` (see
      :func:`_upgrade_data_groups`), and variables referenced by a data group
      role get the ``type`` of the role (and beam fit parameters the
      ``units`` ``"rad"``) when they have none.

    Parameters
    ----------
    xds : xr.Dataset
        Image dataset, for example as read from a zarr store.

    Returns
    -------
    xr.Dataset
        The upgraded dataset. When anything changes this is a shallow copy of
        ``xds`` (the data are not copied and ``xds`` is not modified),
        otherwise ``xds`` itself.
    """
    changes = []
    dataset_attrs = None
    if xds.attrs.get("type") == "image":
        dataset_attrs = {**xds.attrs, "type": "image_dataset"}
        changes.append("dataset type 'image' -> 'image_dataset'")
        # made by make_empty_* of xradio <= 1.2.3, which gave every
        # equatorial frame the equinox J2000
        coordinate_system_info = _upgrade_factory_equinox(
            dataset_attrs.get("coordinate_system_info"), changes
        )
        if coordinate_system_info is not None:
            dataset_attrs["coordinate_system_info"] = coordinate_system_info

    data_groups, var_attr_changes = _upgrade_data_groups(xds, changes)
    if data_groups is not None:
        if dataset_attrs is None:
            dataset_attrs = dict(xds.attrs)
        dataset_attrs["data_groups"] = data_groups

    coord_attrs = {}
    new_coords = {}
    if "frequency" in xds.coords:
        attrs, values, frequency_changes = _upgrade_frequency(
            xds.frequency.attrs, xds.frequency.values
        )
        if frequency_changes:
            coord_attrs["frequency"] = attrs
            changes += frequency_changes
            if values is not None:
                new_coords["frequency"] = ("frequency", values, attrs)

    if "time" in xds.coords:
        attrs, time_changes = _upgrade_time_attrs(xds.time.attrs, "time coordinate")
        if time_changes:
            coord_attrs["time"] = attrs
            changes += time_changes

    for name in ("u", "v"):
        if name in xds.coords:
            attrs = _upgrade_linear_attrs(xds[name].attrs, xds[name].values)
            if attrs is not None:
                coord_attrs[name] = attrs
                changes.append(f"{name} attrs flattened")
            elif not xds[name].attrs.get("units"):
                # add_uv_coordinates of xradio <= 1.2.3, the only producer of
                # u and v without units, gave them in wavelengths
                coord_attrs[name] = {**xds[name].attrs, "units": "lambda"}
                changes.append(f"{name} units 'lambda' added")

    notes = _l_m_attr_notes()
    for name, legacy_note in _LEGACY_L_M_NOTES.items():
        if name in xds.coords and xds[name].attrs.get("note") == legacy_note:
            coord_attrs[name] = {**xds[name].attrs, "note": notes[name]}
            changes.append(f"{name} note updated")

    data_var_attrs = {
        name: {**xds[name].attrs, **updates}
        for name, updates in var_attr_changes.items()
    }
    for name, data_var in xds.data_vars.items():
        obsdate = data_var.attrs.get("obsdate")
        if not isinstance(obsdate, dict) or not isinstance(obsdate.get("attrs"), dict):
            continue
        attrs, time_changes = _upgrade_time_attrs(
            obsdate["attrs"], f"obsdate of {name}"
        )
        if not time_changes:
            continue
        var_attrs = dict(data_var_attrs.get(name, data_var.attrs))
        if attrs is None:
            del var_attrs["obsdate"]
        else:
            var_attrs["obsdate"] = {**obsdate, "attrs": attrs}
        data_var_attrs[name] = var_attrs
        changes += time_changes

    if not changes:
        return xds
    xradio_logger().debug(
        "Upgraded the attributes of an image dataset written by xradio <= "
        f"1.2.3: {'; '.join(changes)}"
    )
    xds = xds.copy(deep=False)
    if new_coords:
        xds = xds.assign_coords(new_coords)
    if dataset_attrs is not None:
        xds.attrs = dataset_attrs
    for name, attrs in coord_attrs.items():
        xds[name].attrs = attrs
    for name, attrs in data_var_attrs.items():
        xds[name].attrs = attrs
    return xds


def _upgrade_factory_equinox(coordinate_system_info, changes: list) -> dict | None:
    """
    Give the fk4 reference direction of a ``make_empty_*`` image of xradio
    <= 1.2.3 its equinox.

    Those factories gave every equatorial frame the equinox ``"j2000.0"``;
    the FK4 frame is the B1950 frame (casacore ``B1950``), as the factories
    and readers write it now.

    Parameters
    ----------
    coordinate_system_info : dict or None
        The dataset's ``coordinate_system_info`` attribute.
    changes : list of str
        Description of the changes (appended to).

    Returns
    -------
    dict or None
        The upgraded ``coordinate_system_info`` (a copy), or None when nothing
        changes.
    """
    if not isinstance(coordinate_system_info, dict):
        return None
    reference = coordinate_system_info.get("reference_direction")
    attrs = reference.get("attrs") if isinstance(reference, dict) else None
    if not isinstance(attrs, dict):
        return None
    if str(attrs.get("frame", "")).lower() != "fk4":
        return None
    if str(attrs.get("equinox", "")).strip().lower() not in ("j2000.0", "j2000"):
        return None
    changes.append("fk4 reference direction equinox 'j2000.0' -> 'b1950.0'")
    return {
        **coordinate_system_info,
        "reference_direction": {
            **reference,
            "attrs": {**attrs, "equinox": "b1950.0"},
        },
    }


def _upgrade_data_groups(xds: xr.Dataset, changes: list) -> tuple[dict | None, dict]:
    """
    Upgrade the data groups of an image dataset of xradio <= 1.2.3.

    * The roles ``model``, ``residual`` and ``dirty`` (the deconvolution
      products of a list of tclean images, which ``open_image`` put in the
      data group of the sky image) become data groups of their own, in which
      the variable is the ``sky`` image (with its ``FLAG_<variable>`` flags,
      if any) and the other images of the group are shared, as
      ``open_image`` builds them now. The variables keep their names.
    * The role ``mask_deconvolve`` becomes ``mask``.
    * Variables referenced by a role that have no ``type`` attribute (or the
      legacy types above) get the type of their role, and beam fit
      parameters without ``units`` get ``"rad"`` (the units the writers
      assume for them).

    Parameters
    ----------
    xds : xr.Dataset
        Image dataset.
    changes : list of str
        Description of the changes (appended to).

    Returns
    -------
    data_groups : dict or None
        The upgraded data groups (a new dict), or None when they do not
        change.
    var_attrs : dict
        Attributes to set, by data variable name.
    """
    groups = xds.attrs.get("data_groups")
    if not isinstance(groups, dict):
        return None, {}
    new_groups = {
        name: dict(group) if isinstance(group, dict) else group
        for name, group in groups.items()
    }
    groups_changed = False
    legacy_types = {*_LEGACY_SKY_ROLES, _LEGACY_MASK_ROLE}
    types = {}
    for name, group in list(new_groups.items()):
        if not isinstance(group, dict):
            continue
        if _LEGACY_MASK_ROLE in group and "mask" not in group:
            group["mask"] = group.pop(_LEGACY_MASK_ROLE)
            groups_changed = True
            changes.append(f"data group {name!r} role 'mask_deconvolve' -> 'mask'")
        for role in _LEGACY_SKY_ROLES:
            variable = group.get(role)
            if not isinstance(variable, str):
                continue
            del group[role]
            groups_changed = True
            if isinstance(new_groups.get(role), dict) and (
                new_groups[role].get("sky") == variable
            ):
                continue
            target = role if role not in new_groups else f"{name}_{role}"
            new_group = {
                r: v
                for r, v in group.items()
                if r not in _SKY_IMAGE_ROLES and r not in _LEGACY_SKY_ROLES
            }
            new_group["sky"] = variable
            if f"FLAG_{variable}" in xds.data_vars:
                new_group["flag"] = f"FLAG_{variable}"
            new_groups[target] = new_group
            types[variable] = "sky"
            changes.append(
                f"data group {name!r} role {role!r} -> sky of data group {target!r}"
            )
    var_attrs = {}
    for group in new_groups.values():
        if not isinstance(group, dict):
            continue
        for role, variable in group.items():
            if (
                role in _TEXT_ROLES
                or not isinstance(variable, str)
                or variable not in xds.data_vars
            ):
                continue
            attrs = xds[variable].attrs
            current = attrs.get("type")
            if variable in types and current in legacy_types:
                var_attrs.setdefault(variable, {})["type"] = types[variable]
            elif current is None or current == _LEGACY_MASK_ROLE:
                var_attrs.setdefault(variable, {}).setdefault("type", role)
            if role.startswith("beam_fit_params") and not attrs.get("units"):
                var_attrs.setdefault(variable, {})["units"] = "rad"
    for variable, updates in var_attrs.items():
        changes.append(f"{variable} attrs {sorted(updates)} added or upgraded")
    return (new_groups if groups_changed else None), var_attrs


def _normalized_observer(observer):
    """
    Return the schema observer for a spectral frame name, or the name itself
    when it has no casacore equivalent (for example ``"icrs"``).
    """
    try:
        return spectral_frame_to_observer(observer)
    except ValueError:
        return observer


def _upgrade_frequency(
    attrs: dict, values: np.ndarray
) -> tuple[dict, np.ndarray | None, list[str]]:
    """
    Upgrade the attributes (and the units of the values) of a frequency
    coordinate.

    Parameters
    ----------
    attrs : dict
        Frequency coordinate attributes.
    values : np.ndarray
        Frequency coordinate values.

    Returns
    -------
    attrs : dict
        The upgraded attributes (a copy).
    values : np.ndarray or None
        The values converted to Hz, or None when they are already in Hz.
    changes : list of str
        Description of the changes, empty when nothing changed.
    """
    attrs = copy.deepcopy(dict(attrs))
    changes = []
    new_values = None

    units = attrs.get("units")
    if units is None:
        attrs["units"] = "Hz"
        changes.append("frequency units 'Hz' added")
    elif units != "Hz":
        try:
            factor = _hz_per_unit(units)
        except ValueError:
            factor = None
        if factor is not None:
            new_values = np.asarray(values, dtype=np.float64) * factor
            attrs["units"] = "Hz"
            changes.append(f"frequency values converted from {units} to Hz")
    for key in _FREQUENCY_QUANTITY_ATTRS:
        quantity = attrs.get(key)
        if not isinstance(quantity, dict) or not isinstance(
            quantity.get("attrs"), dict
        ):
            continue
        q_units = quantity["attrs"].get("units")
        if q_units in (None, "Hz"):
            continue
        try:
            factor = _hz_per_unit(q_units)
        except ValueError:
            continue
        quantity["data"] = (
            np.asarray(quantity["data"], dtype=np.float64) * factor
        ).tolist()
        quantity["attrs"]["units"] = "Hz"
        changes.append(f"frequency {key} converted from {q_units} to Hz")

    reference = attrs.get("reference_frequency")
    observer = None
    if isinstance(reference, dict) and isinstance(reference.get("attrs"), dict):
        observer = reference["attrs"].get("observer")
        if observer is not None:
            normalized = _normalized_observer(observer)
            if normalized != observer:
                reference["attrs"]["observer"] = normalized
                changes.append(
                    f"reference_frequency observer {observer!r} -> {normalized!r}"
                )
                observer = normalized
    if attrs.get("observer") is not None:
        normalized = _normalized_observer(attrs["observer"])
        if normalized != attrs["observer"]:
            changes.append(
                f"frequency observer {attrs['observer']!r} -> {normalized!r}"
            )
            attrs["observer"] = normalized
        if observer is None:
            observer = attrs["observer"]

    frame = attrs.get("frame")
    if frame is None and observer is not None:
        try:
            attrs["frame"] = normalize_spectral_frame(observer)
            changes.append(f"frequency frame {attrs['frame']!r} added")
        except ValueError:
            xradio_logger().warning(
                f"The spectral reference frame (observer {observer!r}) of an "
                "image written by xradio <= 1.2.3 has no casacore equivalent: "
                "its frequency coordinate is left without a frame"
            )
    elif frame is not None:
        try:
            normalized = normalize_spectral_frame(frame)
        except ValueError:
            normalized = frame
        if normalized != frame:
            attrs["frame"] = normalized
            changes.append(f"frequency frame {frame!r} -> {normalized!r}")
    return attrs, new_values, changes


def _upgrade_time_attrs(attrs: dict, what: str) -> tuple[dict | None, list[str]]:
    """
    Upgrade the attributes of a time coordinate or time measure.

    Parameters
    ----------
    attrs : dict
        Time attributes (``units``, ``format``, ``scale``, ``type``).
    what : str
        What the attributes belong to, for messages.

    Returns
    -------
    attrs : dict or None
        The upgraded attributes (a copy). None for an ``obsdate`` whose scale
        has no astropy equivalent (it is to be dropped).
    changes : list of str
        Description of the changes, empty when nothing changed.
    """
    attrs = copy.deepcopy(dict(attrs))
    changes = []
    if "type" not in attrs:
        attrs["type"] = "time"
        changes.append(f"{what} type 'time' added")
    units = attrs.get("units")
    if isinstance(units, list | tuple) and len(units) > 0:
        attrs["units"] = units = units[0]
        changes.append(f"{what} units {units!r} unwrapped from a list")
    time_format = attrs.get("format")
    if not time_format and units in (None, "d"):
        # casacore epochs count from MJD 0
        attrs["format"] = "mjd"
        changes.append(f"{what} format 'mjd' added")
    elif isinstance(time_format, str) and time_format != time_format.lower():
        attrs["format"] = time_format.lower()
        changes.append(f"{what} format {time_format!r} lowercased")
    scale = attrs.get("scale")
    if isinstance(scale, str) and scale.lower() not in _ASTROPY_TIME_SCALES:
        astropy_scale = CASACORE_EPOCH_REF_TO_SCALE.get(scale.upper())
        if astropy_scale is not None:
            attrs["scale"] = astropy_scale
            changes.append(f"{what} scale {scale!r} -> {astropy_scale!r}")
        elif what.startswith("obsdate"):
            changes.append(f"{what} with the scale {scale!r} dropped")
            xradio_logger().warning(
                f"The {what} of an image written by xradio <= 1.2.3 has the "
                f"time scale {scale!r} (a sidereal or unset casacore epoch), "
                "which has no astropy equivalent: it is dropped"
            )
            return None, changes
        else:
            attrs["scale"] = "utc"
            changes.append(f"{what} scale {scale!r} -> 'utc'")
            xradio_logger().warning(
                f"The {what} of an image written by xradio <= 1.2.3 has the "
                f"time scale {scale!r} (a sidereal or unset casacore epoch), "
                "which has no astropy equivalent: it is labelled 'utc'"
            )
    elif isinstance(scale, str) and scale != scale.lower():
        attrs["scale"] = scale.lower()
        changes.append(f"{what} scale {scale!r} lowercased")
    return attrs, changes


def _upgrade_linear_attrs(attrs: dict, values: np.ndarray) -> dict | None:
    """
    Flatten the quantity dictionary that described a ``u`` or ``v`` coordinate.

    Parameters
    ----------
    attrs : dict
        Coordinate attributes.
    values : np.ndarray
        Coordinate values.

    Returns
    -------
    dict or None
        The flat ``units``, ``crval``, ``cdelt`` (for more than one value)
        and ``type`` attributes, or None when ``attrs`` is not a quantity
        dictionary (nothing to upgrade).
    """
    nested = attrs.get("attrs")
    if "units" in attrs or not isinstance(nested, dict) or "units" not in nested:
        return None
    flat = {k: v for k, v in attrs.items() if k not in ("attrs", "data", "dims")}
    flat["units"] = nested["units"]
    data = np.asarray(attrs.get("data", 0.0), dtype=np.float64).ravel()
    flat["crval"] = float(data[0]) if data.size else 0.0
    values = np.asarray(values, dtype=np.float64)
    if values.size >= 2:
        flat["cdelt"] = float(values[1] - values[0])
    flat["type"] = nested.get("type", "quantity")
    return flat
