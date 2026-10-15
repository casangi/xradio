from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import xarray as xr
from xarray.backends import BackendEntrypoint

from xradio._utils.logging import xradio_logger


class ASDMBackendEntryPoint(BackendEntrypoint):
    """Xarray Backend for ALMA/VLA ASDM data."""

    description: ClassVar[str] = (
        "Backend to open ASDM data as Xarray DataTrees, using the data schema of XRADIO ProcessingSet/MeasurementSet v4 "
    )

    url: ClassVar[str] = "https://xradio.readthedocs.io/en/"

    supports_groups: ClassVar[bool] = True

    def _try_lazy_import_open_asdm(self):
        try:
            from xradio.measurement_set._utils._asdm.open_asdm import open_asdm
        except ModuleNotFoundError as exc:
            raise ImportError(
                "The ASDM backend of xradio (xradio_asdm) requires the optional "
                f" dependency pyasdm. It does not seem to be available. Details: {exc}"
            ) from exc
        return open_asdm

    def open_datatree(
        self,
        filename_or_obj,
        *,
        drop_variables: str | Iterable[str] | None = None,
        partition_scheme: list[str] = None,
        include_processor_types: list[str] = None,
        include_spectral_resolution_types: list[str] = None,
        with_pointing: bool = None,
        pointing_for_only_spectral_resolution_types: list[str] = None,
    ) -> xr.DataTree:
        open_asdm = self._try_lazy_import_open_asdm()
        if drop_variables is not None:
            raise RuntimeError("drop_variables not supported")

        return open_asdm(
            filename_or_obj,
            partition_scheme,
            include_processor_types,
            include_spectral_resolution_types,
            with_pointing,
            pointing_for_only_spectral_resolution_types,
        )

    def open_groups_as_dict(
        self,
        filename_or_obj,
        *,
        drop_variables: str | Iterable[str] | None = None,
        **kwargs,
    ):
        asdm_datatree = self.open_datatree(
            filename_or_obj, drop_variables=drop_variables, **kwargs
        )
        return asdm_datatree.to_dict()

    def guess_can_open(self, filename_or_obj) -> bool:
        try:
            filepath = Path(filename_or_obj)
        except Exception as exc:
            xradio_logger().info(
                "Failed to guess if can open (supposed) path: (with type "
                f"{type(filename_or_obj)=}), {filename_or_obj=}. Exception: {exc}"
            )
            return False

        return (
            filepath.is_dir()
            and (filepath / "ASDM.xml").is_file()
            and (filepath / "ASDMBinary").is_dir()
        )
