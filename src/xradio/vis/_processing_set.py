import pandas as pd
from xradio._utils.list_and_array import to_list
import numbers


class processing_set(dict):
    """
    A dictionary subclass representing a Processing Set (PS) that is a set of Measurement Sets v4 (MS).

    This class extends the built-in `dict` class and provides additional methods for manipulating and selecting subsets of the Processing Set.

    Attributes:
        meta (dict): A dictionary containing metadata information about the Processing Set.

    Methods:
        summary(data_group="base"): Returns a summary of the Processing Set as a Pandas table.
        get_ps_max_dims(): Returns the maximum dimension of all the MSs in the Processing Set.
        get_ps_freq_axis(): Combines the frequency axis of all MSs.
        sel(query:str=None, **kwargs): Selects a subset of the Processing Set based on column names and values or a Pandas query.
        ms_sel(**kwargs): Selects a subset of the Processing Set by applying the `sel` method to each individual MS.
        ms_isel(**kwargs): Selects a subset of the Processing Set by applying the `isel` method to each individual MS.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = {"summary": {}}

    def summary(self, data_group="base"):
        """
        Returns a summary of the Processing Set as a Pandas table.

        Args:
            data_group (str): The data group to summarize. Default is "base".

        Returns:
            pandas.DataFrame: A DataFrame containing the summary information.
        """
        if data_group in self.meta["summary"]:
            return self.meta["summary"][data_group]
        else:
            self.meta["summary"][data_group] = self._summary(data_group).sort_values(
                by=["name"], ascending=True
            )
            return self.meta["summary"][data_group]

    def get_ps_max_dims(self):
        """
        Returns the maximum dimension of all the MSs in the Processing Set.

        For example, if the Processing Set contains two MSs with dimensions (50, 20, 30) and (10, 30, 40), the maximum dimensions will be (50, 30, 40).

        Returns:
            dict: A dictionary containing the maximum dimensions of the Processing Set.
        """
        if "max_dims" in self.meta:
            return self.meta["max_dims"]
        else:
            self.meta["max_dims"] = self._get_ps_max_dims()
            return self.meta["max_dims"]

    def get_ps_freq_axis(self):
        """
        Combines the frequency axis of all MSs.

        Returns:
            xarray.DataArray: The frequency axis of the Processing Set.
        """
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
            "scan_number": [],
            "spw_name": [],
            # "field_id": [],
            "field_name": [],
            # "source_id": [],
            "source_name": [],
            # "num_lines": [],
            "line_name": [],
            "field_coords": [],
            "start_frequency": [],
            "end_frequency": [],
        }
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        for key, value in self.items():
            summary_data["name"].append(key)
            summary_data["obs_mode"].append(value.attrs["partition_info"]["obs_mode"])
            summary_data["spw_name"].append(
                value.attrs["partition_info"]["spectral_window_name"]
            )
            summary_data["polarization"].append(value.polarization.values)
            summary_data["scan_number"].append(
                value.attrs["partition_info"]["scan_number"]
            )

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

            summary_data["line_name"].append(value.attrs["partition_info"]["line_name"])

            # summary_data["num_lines"].append(value.attrs["partition_info"]["num_lines"])
            summary_data["start_frequency"].append(
                to_list(value["frequency"].values)[0]
            )
            summary_data["end_frequency"].append(to_list(value["frequency"].values)[-1])

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
            ), "Frequency reference frame not consistent in Processing Set."
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

    def sel(self, string_exact_match: bool = True, query: str = None, **kwargs):
        """
        Selects a subset of the Processing Set based on column names and values or a Pandas query.

        The following columns are supported: name, obs_mode, polarization, spw_name, field_name, source_name, field_coords, start_frequency, end_frequency.

        This function will not apply any selection on the MS data so data will not be dropped for example if a MS has field_name=['field_0','field_10','field_08'] and ps.sel(field_name='field_0') is done the resulting MS will still have field_name=['field_0','field_10','field_08'].

        Examples:
            ps.sel(obs_mode='OBSERVE_TARGET#ON_SOURCE', polarization=['RR', 'LL']) # Select all MSs with obs_mode 'OBSERVE_TARGET#ON_SOURCE' and polarization 'RR' or 'LL'.
            ps.sel(query='start_frequency > 100e9 AND end_frequency < 200e9') # Select all MSs with start_frequency greater than 100 GHz and less than 200 GHz.

        Args:
            query (str): A Pandas query string. Default is None.
            string_exact_match (bool): If True, the selection will be an exact match for string and string list columns. Default is True.
            **kwargs: Keyword arguments representing column names and values to filter the Processing Set.

        Returns:
            processing_set: The subset of the Processing Set.
        """
        import numpy as np

        # def select_rows(df, col, input_strings):
        #     return df[df[col].apply(lambda x: any(i in x for i in input_strings))]

        # def select_rows(df, col, sel, string_exact_match):
        #     def check_selection(row_val):
        #         if isinstance(row_val, numbers.Number) or string_exact_match:
        #             return any(i == row_val for i in sel) #If values are numbers
        #         return any(i in row_val for i in sel) #If values are strings
        #     return df[df[col].apply(check_selection)]

        def select_rows(df, col, sel_vals, string_exact_match):
            def check_selection(row_val):
                row_val = to_list(
                    row_val
                )  # make sure that it is a list so that we can iterate over it.

                for rw in row_val:
                    for s in sel_vals:
                        if string_exact_match:
                            if rw == s:
                                return True
                        else:
                            if s in rw:
                                return True
                return False

            return df[df[col].apply(check_selection)]

        summary_table = self.summary()
        for key, value in kwargs.items():
            value = to_list(value)  # make sure value is a list.

            if len(value) == 1 and isinstance(value[0], slice):
                summary_table = summary_table[
                    summary_table[key].between(value[0].start, value[0].stop)
                ]
            else:
                summary_table = select_rows(
                    summary_table, key, value, string_exact_match
                )

        if query is not None:
            summary_table = summary_table.query(query)

        sub_ps = processing_set()
        for key, val in self.items():
            if key in summary_table["name"].values:
                sub_ps[key] = val

        return sub_ps

    def ms_sel(self, **kwargs):
        """
        Selects a subset of the Processing Set by applying the `sel` method to each MS.

        Args:
            **kwargs: Keyword arguments representing column names and values to filter the Processing Set.

        Returns:
            processing_set: The subset of the Processing Set.
        """
        sub_ps = processing_set()
        for key, val in self.items():
            sub_ps[key] = val.sel(kwargs)
        return sub_ps

    def ms_isel(self, **kwargs):
        """
        Selects a subset of the Processing Set by applying the `isel` method to each MS.

        Args:
            **kwargs: Keyword arguments representing dimension names and indices to select from the Processing Set.

        Returns:
            processing_set: The subset of the Processing Set.
        """
        sub_ps = processing_set()
        for key, val in self.items():
            sub_ps[key] = val.isel(kwargs)
        return sub_ps
