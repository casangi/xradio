"""xarray backend engine for reading MSv2 as MSv4-schema datasets.

Registers ``xradio:msv2`` so that users can write::

    xr.open_datatree("path/to/ms.ms", engine="xradio:msv2")

and get back the same DataTree that :func:`open_msv2` produces.
"""

import os

import xarray as xr
from xarray.backends import BackendEntrypoint


class MSv2BackendEntrypoint(BackendEntrypoint):
    """xarray backend for CASA MSv2 tables via xradio."""

    def open_datatree(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        partition_scheme=None,
        main_chunksize=None,
        with_pointing=True,
        pointing_interpolate=False,
        ephemeris_interpolate=False,
        phase_cal_interpolate=False,
        sys_cal_interpolate=False,
    ) -> xr.DataTree:
        from xradio.measurement_set.open_msv2 import open_msv2

        return open_msv2(
            str(filename_or_obj),
            partition_scheme=partition_scheme,
            main_chunksize=main_chunksize,
            with_pointing=with_pointing,
            pointing_interpolate=pointing_interpolate,
            ephemeris_interpolate=ephemeris_interpolate,
            phase_cal_interpolate=phase_cal_interpolate,
            sys_cal_interpolate=sys_cal_interpolate,
        )

    def guess_can_open(self, filename_or_obj) -> bool:
        try:
            path = str(filename_or_obj)
        except Exception:
            return False
        return os.path.isdir(path) and os.path.isfile(os.path.join(path, "table.dat"))

    description = "Open CASA MSv2 tables as MSv4-schema DataTree via xradio"
    open_dataset_parameters = [
        "filename_or_obj",
        "drop_variables",
        "partition_scheme",
        "main_chunksize",
        "with_pointing",
        "pointing_interpolate",
        "ephemeris_interpolate",
        "phase_cal_interpolate",
        "sys_cal_interpolate",
    ]
