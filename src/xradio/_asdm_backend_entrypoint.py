"""
Xarray backend entry point ``xradio_asdm`` (ASDM -> processing set of MSv4s).

This module is loaded by xarray whenever it lists its engines (for example on
every ``open_dataset`` / ``open_datatree`` call without an explicit built-in
engine), for every installation of xradio. It must therefore stay light: it only
imports the standard library and ``xarray.backends``. The ASDM reader (and its
optional dependency pyasdm, from the ``asdm`` extra) is imported only when an
ASDM is actually opened.
"""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from xarray.backends import BackendEntrypoint

_INSTALL_HINT = "pip install 'xradio[asdm]'"


class ASDMBackendEntryPoint(BackendEntrypoint):
    """Xarray Backend for ALMA/VLA ASDM data.

    Opens an ASDM directory as a processing set DataTree of MSv4s, with::

        xr.open_datatree(asdm_path, engine="xradio_asdm", ...)

    See :func:`xradio.measurement_set.open_asdm` for the backend options
    (partition_scheme, include_processor_types, etc.).
    Requires the optional dependency pyasdm (``pip install 'xradio[asdm]'``).
    """

    description: ClassVar[str] = (
        "Backend to open ASDM data as Xarray DataTrees, using the data schema of "
        "XRADIO ProcessingSet/MeasurementSet v4"
    )

    url: ClassVar[str] = (
        "https://xradio.readthedocs.io/en/latest/measurement_set/api.html"
        "#asdm-xarray-engine-xradio-asdm"
    )

    # Lets xr.open_datatree auto-detect ASDM directories (guess_can_open)
    supports_groups: ClassVar[bool] = True

    open_dataset_parameters: ClassVar[tuple[str, ...]] = (
        "filename_or_obj",
        "drop_variables",
    )

    def open_dataset(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
    ):
        raise NotImplementedError(
            "An ASDM is opened as a DataTree (processing set) of MSv4s, use "
            "xarray.open_datatree(path, engine='xradio_asdm') instead of "
            "open_dataset."
        )

    def open_datatree(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        partition_scheme: list[str] | None = None,
        include_processor_types: list[str] | None = None,
        include_spectral_resolution_types: list[str] | None = None,
        with_pointing: bool = False,
        pointing_for_only_spectral_resolution_types: list[str] | None = None,
    ):
        """Open an ASDM as a processing set DataTree (see ``open_asdm``)."""
        if drop_variables is not None:
            raise RuntimeError("drop_variables not supported")

        open_asdm = _import_open_asdm()
        return open_asdm(
            filename_or_obj,
            partition_scheme=partition_scheme,
            include_processor_types=include_processor_types,
            include_spectral_resolution_types=include_spectral_resolution_types,
            with_pointing=bool(with_pointing),
            pointing_for_only_spectral_resolution_types=(
                pointing_for_only_spectral_resolution_types
            ),
        )

    def open_groups_as_dict(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        **kwargs,
    ) -> dict:
        """Open an ASDM as a dict {group path: Dataset} (see ``open_datatree``)."""
        return self.open_datatree(
            filename_or_obj, drop_variables=drop_variables, **kwargs
        ).to_dict()

    def guess_can_open(self, filename_or_obj: Any) -> bool:
        """True for a path (str or os.PathLike) to an ASDM directory: a directory
        with an ASDM.xml file and an ASDMBinary sub-directory."""
        if not isinstance(filename_or_obj, str | os.PathLike):
            return False
        try:
            path = Path(os.fspath(filename_or_obj)).expanduser()
            return bool(
                path.is_dir()
                and (path / "ASDM.xml").is_file()
                and (path / "ASDMBinary").is_dir()
            )
        except (OSError, TypeError, ValueError):
            return False


def _import_open_asdm():
    """Import the ASDM reader, with an install hint if its dependencies are missing."""
    try:
        from xradio.measurement_set._utils._asdm.open_asdm import open_asdm
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "xradio":
            raise
        raise ImportError(
            "The xradio_asdm backend requires optional dependencies that are not "
            f"installed (missing module: {exc.name!r}). Install them with: "
            f"{_INSTALL_HINT}"
        ) from exc
    return open_asdm
