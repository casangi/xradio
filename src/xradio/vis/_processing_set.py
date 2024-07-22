import pandas as pd


class processing_set(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = {"summary": {}}

    def summary(self, data_group="base"):
        if data_group in self.meta["summary"]:
            return self.meta["summary"][data_group]
        else:
            self.meta["summary"][data_group] = self._summary(data_group)
            return self.meta["summary"][data_group]

    def get_ps_max_dims(self):
        if "max_dims" in self.meta:
            return self.meta["max_dims"]
        else:
            self.meta["max_dims"] = self._get_ps_max_dims()
            return self.meta["max_dims"]

    def get_ps_freq_axis(self):
        if "freq_axis" in self.meta:
            return self.meta["freq_axis"]
        else:
            self.meta["freq_axis"] = self._get_ps_freq_axis()
            return self.meta["freq_axis"]

    def _summary(self, data_group="base"):
        summary_data = {
            "name": [],
            "obs_mode": [],
            "shape": [],
            "polarization": [],
            "spw_id": [],
            # "field_id": [],
            "field_name": [],
            # "source_id": [],
            "source_name": [],
            "field_coords": [],
            "start_frequency": [],
            "end_frequency": [],
        }
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        for key, value in self.items():
            summary_data["name"].append(key)
            summary_data["obs_mode"].append(value.attrs["partition_info"]["obs_mode"])
            summary_data["spw_id"].append(
                value.attrs["partition_info"]["spectral_window_id"]
            )
            summary_data["polarization"].append(value.polarization.values)

            if "visibility" in value.attrs["data_groups"][data_group]:
                data_name = value.attrs["data_groups"][data_group]["visibility"]
                center_name = "FIELD_PHASE_CENTER"

            if "spectrum" in value.attrs["data_groups"][data_group]:
                data_name = value.attrs["data_groups"][data_group]["spectrum"]
                center_name = "FIELD_REFERENCE_CENTER"

            summary_data["shape"].append(value[data_name].shape)

            # summary_data["field_id"].append(value.attrs["partition_info"]["field_id"])
            # summary_data["source_id"].append(value.attrs["partition_info"]["source_id"])

            summary_data["field_name"].append(
                value.attrs["partition_info"]["field_name"]
            )
            summary_data["source_name"].append(
                value.attrs["partition_info"]["source_name"]
            )
            summary_data["start_frequency"].append(value["frequency"].values[0])
            summary_data["end_frequency"].append(value["frequency"].values[-1])

            if value[data_name].attrs["field_and_source_xds"].is_ephemeris:
                summary_data["field_coords"].append("Ephemeris")
            elif (
                "time"
                in value[data_name].attrs["field_and_source_xds"][center_name].coords
            ):
                summary_data["field_coords"].append("Multi-Phase-Center")
            else:
                ra_dec_rad = (
                    value[data_name].attrs["field_and_source_xds"][center_name].values
                )
                frame = (
                    value[data_name]
                    .attrs["field_and_source_xds"][center_name]
                    .attrs["frame"]
                    .lower()
                )

                coord = SkyCoord(
                    ra=ra_dec_rad[0] * u.rad, dec=ra_dec_rad[1] * u.rad, frame=frame
                )

                summary_data["field_coords"].append(
                    [
                        frame,
                        coord.ra.to_string(unit=u.hour, precision=2),
                        coord.dec.to_string(unit=u.deg, precision=2),
                    ]
                )

        summary_df = pd.DataFrame(summary_data)
        return summary_df

    def _get_ps_freq_axis(self):
        import xarray as xr

        spw_ids = []
        freq_axis_list = []
        frame = self.get(0).frequency.attrs["frame"]
        for ms_xds in self.values():
            assert (
                frame == ms_xds.frequency.attrs["frame"]
            ), "Frequency reference frame not consistent in processing set."
            if ms_xds.frequency.attrs["spectral_window_id"] not in spw_ids:
                spw_ids.append(ms_xds.frequency.attrs["spectral_window_id"])
                freq_axis_list.append(ms_xds.frequency)

        freq_axis = xr.concat(freq_axis_list, dim="frequency").sortby("frequency")
        return freq_axis

    def _get_ps_max_dims(self):
        max_dims = None
        for ms_xds in self.values():
            if max_dims is None:
                max_dims = dict(ms_xds.sizes)
            else:
                for dim_name, size in ms_xds.sizes.items():
                    if dim_name in max_dims:
                        if max_dims[dim_name] < size:
                            max_dims[dim_name] = size
                    else:
                        max_dims[dim_name] = size
        return max_dims

    def get(self, id):
        return self[list(self.keys())[id]]

    def sel(self, **kwargs):
        import numpy as np

        summary_table = self.summary()
        for key, value in kwargs.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                summary_table = summary_table[summary_table[key].isin(value)]
            elif isinstance(value, slice):
                summary_table = summary_table[
                    summary_table[key].between(value.start, value.stop)
                ]
            else:
                summary_table = summary_table[summary_table[key] == value]

        sub_ps = processing_set()
        for key, val in self.items():
            if key in summary_table["name"].values:
                sub_ps[key] = val

        return sub_ps

    def ms_sel(self, **kwargs):
        sub_ps = processing_set()
        for key, val in self.items():
            sub_ps[key] = val.sel(kwargs)
        return sub_ps

    def ms_isel(self, **kwargs):
        sub_ps = processing_set()
        for key, val in self.items():
            sub_ps[key] = val.isel(kwargs)
        return sub_ps
