"""
Plan and stage the outputs of ``write_image``.

An image dataset can hold several images, for example the sky, point spread
function and primary beam images of several data groups, while a CASA or FITS
image holds a single image. :func:`plan_image_outputs` decides which data
variables are written as separate images, under which data group role, which
variables travel with each image (the flags and beam fit parameters of a sky
image, the beam fit parameters of a point spread function) and the output
paths, so that the CASA and FITS drivers and the overwrite check of
``write_image`` agree.

Output names
------------
Each data variable is written once. When a single image is written it is
named ``<name>``, as given. When several images are written, each is named
``<name>.<g1>.<g2>...<gn>.<role>``, where ``g1`` to ``gn`` are, in data group
order, all data groups whose ``<role>`` refers to the variable. For FITS, a
``.fits`` extension of ``<name>`` (in any case) stays last, so ``img.fits``
gives ``img.base.sky.fits``.

:func:`write_outputs_atomically` writes the outputs into a temporary directory
next to them and moves them into place only after all of them have been
written, so that a failed write leaves nothing behind and a lazily read source
is never removed before it has been read.
"""

import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace

import xarray as xr

from xradio._utils.logging import xradio_logger
from xradio._utils.schema import get_data_group_keys

#: Data group roles whose variables travel with the image of another role,
#: keyed by that role: a sky image carries its flags and beam fit parameters,
#: a point spread function its beam fit parameters.
COMPANION_ROLES = {
    "sky": {"flag": "flag", "beam_fit_params": "beam_fit_params_sky"},
    "point_spread_function": {
        "beam_fit_params": "beam_fit_params_point_spread_function"
    },
}

# Data group entries that hold text, not the name of a data variable
_TEXT_ROLES = ("description", "date")

_FITS_EXTENSION = ".fits"

_FORMAT_NAMES = {"casa": "CASA", "fits": "FITS"}


@dataclass(frozen=True)
class ImageOutput:
    """One image written by ``write_image``.

    Attributes
    ----------
    variable : str
        Name of the data variable written as the image.
    role : str
        Data group role of the variable, for example ``"sky"``.
    groups : tuple of str
        Data groups whose ``role`` refers to the variable, in data group
        order.
    path : str
        Output path.
    plane : str
        ``"sky"`` for an image with l and m dimensions, ``"aperture"`` for
        one with u and v dimensions.
    flag : str or None
        Flag variable written with the image (sky images only).
    beam_fit_params : str or None
        Beam fit parameters variable written with the image (sky and point
        spread function images only).
    """

    variable: str
    role: str
    groups: tuple[str, ...]
    path: str
    plane: str
    flag: str | None = None
    beam_fit_params: str | None = None


@dataclass(frozen=True)
class ImageWritePlan:
    """The images ``write_image`` writes for a dataset in one format.

    Attributes
    ----------
    outputs : tuple of ImageOutput
        The images to write, in data group order.
    warnings : tuple of str
        Messages about data variables that are not written, for the driver
        to log.
    """

    outputs: tuple[ImageOutput, ...]
    warnings: tuple[str, ...] = ()

    @property
    def paths(self) -> list[str]:
        """The output paths, in output order."""
        return [output.path for output in self.outputs]

    def in_directory(self, directory: str) -> "ImageWritePlan":
        """Return the same plan with every output moved into ``directory``.

        Parameters
        ----------
        directory : str
            The directory that receives the outputs (keeping their base
            names), for example the staging directory of
            :func:`write_outputs_atomically`.

        Returns
        -------
        ImageWritePlan
            The plan with the new output paths.
        """
        return replace(
            self,
            outputs=tuple(
                replace(
                    output,
                    path=os.path.join(directory, os.path.basename(output.path)),
                )
                for output in self.outputs
            ),
        )


def _image_roles() -> list[str]:
    """Data group roles whose variables are written as images, in schema
    order."""
    companions = {role for roles in COMPANION_ROLES.values() for role in roles.values()}
    return [
        role
        for role in get_data_group_keys(schema_name="image")
        if role not in companions and role not in _TEXT_ROLES
    ]


def _image_plane(data_array: xr.DataArray, out_format: str) -> str | None:
    """Return the plane of an image variable, or None when ``out_format``
    cannot hold it."""
    dims = set(data_array.dims)
    if {"l", "m"} <= dims:
        return "sky"
    if out_format == "casa" and {"u", "v"} <= dims:
        return "aperture"
    return None


def _check_group_name(group: str) -> None:
    """Raise if a data group name cannot be part of a file name."""
    separators = [sep for sep in (os.sep, os.altsep) if sep]
    if (
        not isinstance(group, str)
        or group in ("", ".", "..")
        or any(sep in group for sep in separators)
    ):
        raise ValueError(
            f"Cannot write the image dataset to several images: the data "
            f"group name {group!r} cannot be part of an output file name"
        )


def _output_path(
    image_store_name: str, groups: tuple[str, ...], role: str, out_format: str
) -> str:
    """Name of one of several outputs: ``<name>.<g1>...<gn>.<role>``, with a
    FITS ``.fits`` extension kept last."""
    stem, extension = image_store_name, ""
    if out_format == "fits" and image_store_name.lower().endswith(_FITS_EXTENSION):
        stem = image_store_name[: -len(_FITS_EXTENSION)]
        extension = image_store_name[-len(_FITS_EXTENSION) :]
    return ".".join([stem, *groups, role]) + extension


def plan_image_outputs(
    xds: xr.Dataset, image_store_name: str, out_format: str
) -> ImageWritePlan:
    """Plan the images that ``write_image`` writes in CASA or FITS format.

    Every data variable that fills an image role (``sky``,
    ``point_spread_function``, ``primary_beam``, ``mask``, ...) in some data
    group is written once, under the first such role in data group order.
    Flags and beam fit parameters are not written as images: they travel with
    the image of their data group (taken from the first data group of the
    image that names them). Images the format cannot hold are skipped: FITS
    holds only sky plane images (l and m dimensions); CASA holds sky plane and
    aperture plane (u and v dimensions) images, but not the normalization
    images without l and m or u and v dimensions.

    Parameters
    ----------
    xds : xr.Dataset
        The image dataset, with a ``data_groups`` attribute.
    image_store_name : str
        The output name given to ``write_image``.
    out_format : str
        ``"casa"`` or ``"fits"``.

    Returns
    -------
    ImageWritePlan
        The outputs and the warnings about variables that are not written.

    Raises
    ------
    ValueError
        If the dataset has no data groups or no image the format can hold, if
        a data group refers to a missing flag or beam fit parameters variable,
        or if a data group name cannot be part of a file name.
    """
    out_format = out_format.lower()
    if out_format not in _FORMAT_NAMES:
        raise ValueError(f"Cannot plan image outputs for format {out_format!r}")
    format_name = _FORMAT_NAMES[out_format]
    data_groups = xds.attrs.get("data_groups")
    if not isinstance(data_groups, dict) or not data_groups:
        raise ValueError(
            f"Cannot write {format_name} images: the dataset has no "
            "data_groups attribute naming its images"
        )
    image_roles = _image_roles()
    known_roles = (
        set(image_roles)
        | {role for roles in COMPANION_ROLES.values() for role in roles.values()}
        | set(_TEXT_ROLES)
    )
    warnings = []

    # The role of each image variable: the first image role that refers to
    # it, in data group order and then schema role order
    variable_roles = {}
    missing = []
    for group_name, group in data_groups.items():
        for role in image_roles:
            variable = group.get(role)
            if variable is None:
                continue
            if not isinstance(variable, str) or variable not in xds.data_vars:
                missing.append(f"{group_name}.{role} = {variable!r}")
                continue
            variable_roles.setdefault(variable, role)
    if missing:
        warnings.append(
            "Data group entries refer to variables that are not in the "
            f"dataset and are not written: {', '.join(missing)}"
        )

    outputs = []
    skipped = []
    for variable, role in variable_roles.items():
        groups = tuple(
            name for name, group in data_groups.items() if group.get(role) == variable
        )
        other_roles = sorted(
            {
                other
                for group in data_groups.values()
                for other in image_roles
                if other != role and group.get(other) == variable
            }
        )
        if other_roles:
            warnings.append(
                f"{variable} fills the roles {role}, {', '.join(other_roles)}; "
                f"it is written once, as {role}"
            )
        plane = _image_plane(xds[variable], out_format)
        if plane is None:
            skipped.append(variable)
            continue
        companions = {}
        for kind, companion_role in COMPANION_ROLES.get(role, {}).items():
            names = [
                data_groups[name][companion_role]
                for name in groups
                if data_groups[name].get(companion_role) is not None
            ]
            if not names:
                continue
            if not isinstance(names[0], str) or names[0] not in xds.data_vars:
                raise ValueError(
                    f"Cannot write {variable}: its data group refers to "
                    f"{names[0]!r} as {companion_role}, but the dataset has no "
                    "such data variable"
                )
            if any(name != names[0] for name in names):
                warnings.append(
                    f"The data groups {', '.join(groups)} of {variable} refer "
                    f"to different {companion_role} variables {sorted(set(names))}; "
                    f"{names[0]} is written with {variable}"
                )
            companions[kind] = names[0]
        outputs.append(
            ImageOutput(
                variable=variable,
                role=role,
                groups=groups,
                path=image_store_name,
                plane=plane,
                **companions,
            )
        )

    if skipped:
        reason = (
            "only sky plane images (with l and m dimensions) can be written to FITS"
            if out_format == "fits"
            else "they have neither l and m nor u and v dimensions"
        )
        warnings.append(
            f"Not writing {', '.join(skipped)} to {format_name} images: {reason}"
        )
    if not outputs:
        detail = f" ({warnings[-1]})" if warnings else ""
        raise ValueError(
            f"No valid image types found in xds to write to {format_name} "
            f"images{detail}."
        )

    carried = {
        name
        for output in outputs
        for name in (output.flag, output.beam_fit_params)
        if name is not None
    }
    written = {output.variable for output in outputs} | carried | set(skipped)
    unwritten = [name for name in xds.data_vars if name not in written]
    if unwritten:
        unknown_roles = sorted(
            {
                role
                for group in data_groups.values()
                for role, name in group.items()
                if role not in known_roles and name in unwritten
            }
        )
        detail = (
            f" (data group roles {', '.join(unknown_roles)} are not image roles)"
            if unknown_roles
            else ""
        )
        warnings.append(
            f"Not writing {', '.join(unwritten)} to {format_name} images: they "
            f"are neither an image of a data group nor the flags or beam fit "
            f"parameters of one{detail}"
        )

    if len(outputs) > 1:
        for output in outputs:
            for group in output.groups:
                _check_group_name(group)
        outputs = [
            replace(
                output,
                path=_output_path(
                    image_store_name, output.groups, output.role, out_format
                ),
            )
            for output in outputs
        ]
        # names that differ only in case are the same file on case
        # insensitive file systems (the default on macOS)
        keys = [output.path.casefold() for output in outputs]
        duplicates = sorted(
            {
                output.path
                for output, key in zip(outputs, keys, strict=True)
                if keys.count(key) > 1
            }
        )
        if duplicates:
            raise ValueError(
                f"Cannot write the image dataset: several images would be "
                f"written to {', '.join(duplicates)} (data group names that "
                "differ only in case, or that contain '.', can make output "
                "names ambiguous)"
            )
    return ImageWritePlan(outputs=tuple(outputs), warnings=tuple(warnings))


@contextmanager
def naming_the_variable(
    variable: str, format_name: str, internal_names: Sequence[str]
) -> Iterator[None]:
    """Make the errors raised while writing one image name its data variable.

    The CASA and FITS drivers write every image as a single image dataset in
    which the image has an internal name (SKY or APERTURE). Errors are
    re-raised (as the same type) with a message that names the variable of
    the input dataset instead: a message that starts with "Cannot write
    <internal name> " gets the variable in place of the internal name, one
    about a part of the image ("Cannot write <part> to <format>...") gets
    " of <variable>" after the part, and any other message the prefix
    "Cannot write <variable> to <format>: ". Errors of other types get that
    prefix as a note.

    Parameters
    ----------
    variable : str
        Name of the image's data variable in the dataset given to
        ``write_image``.
    format_name : str
        ``"CASA"`` or ``"FITS"``.
    internal_names : sequence of str
        Names the writer's messages use for the image, for example ``"SKY"``
        or ``"the image"``.
    """
    try:
        yield
    except (RuntimeError, TypeError, ValueError) as exc:
        if type(exc) not in (RuntimeError, TypeError, ValueError):
            # subclasses may need other constructor arguments
            exc.add_note(f"Cannot write {variable} to {format_name}")
            raise
        message = str(exc)
        part = re.match(
            rf"Cannot write (.+?) (to (?:a )?{format_name}\b.*)", message, re.DOTALL
        )
        for name in internal_names:
            prefix = f"Cannot write {name} "
            if message.startswith(prefix):
                message = f"Cannot write {variable} {message[len(prefix) :]}"
                break
        else:
            if part is not None:
                message = f"Cannot write {part[1]} of {variable} {part[2]}"
            else:
                message = f"Cannot write {variable} to {format_name}: {message}"
        raise type(exc)(message) from exc
    except Exception as exc:
        exc.add_note(f"Cannot write {variable} to {format_name}")
        raise


def check_overwrite(paths: Sequence[str], overwrite: bool) -> None:
    """Raise if an output exists and may not be replaced.

    Parameters
    ----------
    paths : sequence of str
        Output paths.
    overwrite : bool
        Whether existing outputs may be replaced.

    Raises
    ------
    FileExistsError
        If ``overwrite`` is False and any output path exists.
    """
    existing = [path for path in paths if os.path.lexists(path)]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output path{'s' if len(existing) > 1 else ''} "
            f"{', '.join(existing)} already exist{'' if len(existing) > 1 else 's'}. "
            "Set overwrite=True to replace existing outputs; nothing has been "
            "written."
        )


def _create_missing_directories(directory: str) -> list[str]:
    """Create ``directory`` and its missing parents; return the directories
    created, deepest first."""
    created = []
    current = os.path.abspath(directory)
    while not os.path.exists(current):
        created.append(current)
        current = os.path.dirname(current)
    if created:
        os.makedirs(directory, exist_ok=True)
    return created


def _remove_directories(directories: list[str]) -> None:
    """Remove directories created by :func:`_create_missing_directories`,
    deepest first, stopping at the first one that is not empty."""
    for directory in directories:
        try:
            os.rmdir(directory)
        except OSError:
            break


class _RollbackError(OSError):
    """Raised when existing outputs could not be restored after a failed
    move; the staging directory, which holds them, is then kept."""


def _move_into_place(paths: Sequence[str], staging: str, overwrite: bool) -> None:
    """Move the staged outputs into place, replacing existing outputs.

    Existing outputs are first moved into a directory inside ``staging``, so
    that they can be restored if a later move fails.
    """
    backup_dir = None
    replaced = []
    moved = []
    try:
        for path in paths:
            staged = os.path.join(staging, os.path.basename(path))
            if os.path.lexists(path):
                if not overwrite:
                    raise FileExistsError(
                        f"Output path {path} was created while the image was "
                        "being written; it has not been replaced (overwrite=False)"
                    )
                if backup_dir is None:
                    backup_dir = tempfile.mkdtemp(prefix=".replaced.", dir=staging)
                xradio_logger().warning(
                    f"Because overwrite=True, replacing existing path {path}"
                )
                backup = os.path.join(backup_dir, os.path.basename(path))
                os.rename(path, backup)
                replaced.append((path, backup))
            os.rename(staged, path)
            moved.append((path, staged))
    except BaseException:
        try:
            for path, staged in reversed(moved):
                os.rename(path, staged)
            for path, backup in reversed(replaced):
                os.rename(backup, path)
        except OSError as exc:
            raise _RollbackError(
                "Writing the image failed and the outputs could not all be "
                f"moved back; the new and the replaced outputs are kept in {staging}"
            ) from exc
        raise


def write_outputs_atomically(
    paths: Sequence[str], overwrite: bool, write: Callable[[str], None]
) -> None:
    """Write outputs into a temporary directory, then move them into place.

    The outputs are written by ``write`` into a new temporary directory next
    to them (in the same directory, so that moving them is a rename). Only
    when ``write`` returns are they moved to ``paths``, replacing any existing
    outputs. If anything fails, the temporary directory is removed and
    nothing at ``paths`` changes, so a dataset that is read lazily from one of
    the outputs (for example an image opened, edited and written back to the
    same path) is read completely before it is replaced.

    Parameters
    ----------
    paths : sequence of str
        Output paths, all in the same directory. Missing parent directories
        are created (and removed again if the write fails).
    overwrite : bool
        Whether existing outputs are replaced. When False and an output
        exists, FileExistsError is raised before anything is written.
    write : callable
        ``write(directory)`` writes every output into ``directory``, under
        its base name.

    Raises
    ------
    FileExistsError
        If an output exists and ``overwrite`` is False.
    """
    check_overwrite(paths, overwrite)
    parents = {os.path.dirname(path) for path in paths}
    if len(parents) != 1:
        raise ValueError(f"Outputs {list(paths)} are not in one directory")
    parent = parents.pop()
    created = _create_missing_directories(parent) if parent else []
    try:
        # a hidden name that shows which image is being written (shortened
        # to stay within file name length limits); absolute, so that dask
        # workers with another working directory write to the same place
        staging = tempfile.mkdtemp(
            prefix=f".{os.path.basename(paths[0])[:100]}.",
            suffix=".writing",
            dir=os.path.abspath(parent or "."),
        )
    except BaseException:
        _remove_directories(created)
        raise
    try:
        write(staging)
        _move_into_place(paths, staging, overwrite)
    except _RollbackError:
        raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        _remove_directories(created)
        raise
    try:
        shutil.rmtree(staging)
    except OSError as exc:
        xradio_logger().warning(
            f"The image was written, but its temporary directory {staging} "
            f"could not be removed: {exc}"
        )
