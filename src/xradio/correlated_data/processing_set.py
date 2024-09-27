import pandas as pd
from xradio._utils.list_and_array import to_list
import numbers
import numpy as np
import toolviper.utils.logger as logger
import xarray as xr


class ProcessingSet(dict):
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
            "intents": [],
            "shape": [],
            "polarization": [],
            "scan_number": [],
            "spw_name": [],
            "field_name": [],
            "source_name": [],
            "line_name": [],
            "field_coords": [],
            "start_frequency": [],
            "end_frequency": [],
        }
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        for key, value in self.items():
            summary_data["name"].append(key)
            summary_data["intents"].append(value.attrs["partition_info"]["intents"])
            summary_data["spw_name"].append(
                value.attrs["partition_info"]["spectral_window_name"]
            )
            summary_data["polarization"].append(value.polarization.values)
            summary_data["scan_number"].append(
                value.attrs["partition_info"]["scan_number"]
            )
            data_name = value.attrs["data_groups"][data_group]["correlated_data"]

            if "VISIBILITY" in data_name:
                center_name = "FIELD_PHASE_CENTER"

            if "SPECTRUM" in data_name:
                center_name = "FIELD_REFERENCE_CENTER"

            summary_data["shape"].append(value[data_name].shape)

            summary_data["field_name"].append(
                value.attrs["partition_info"]["field_name"]
            )
            summary_data["source_name"].append(
                value.attrs["partition_info"]["source_name"]
            )

            summary_data["line_name"].append(value.attrs["partition_info"]["line_name"])

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

        spw_ids = []
        freq_axis_list = []
        frame = self.get(0).frequency.attrs["frame"]
        for cor_xds in self.values():
            assert (
                frame == cor_xds.frequency.attrs["frame"]
            ), "Frequency reference frame not consistent in Processing Set."
            if cor_xds.frequency.attrs["spectral_window_id"] not in spw_ids:
                spw_ids.append(cor_xds.frequency.attrs["spectral_window_id"])
                freq_axis_list.append(cor_xds.frequency)

        freq_axis = xr.concat(freq_axis_list, dim="frequency").sortby("frequency")
        return freq_axis

    def _get_ps_max_dims(self):
        max_dims = None
        for cor_xds in self.values():
            if max_dims is None:
                max_dims = dict(cor_xds.sizes)
            else:
                for dim_name, size in cor_xds.sizes.items():
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

        The following columns are supported: name, intents, polarization, spw_name, field_name, source_name, field_coords, start_frequency, end_frequency.

        This function will not apply any selection on the MS data so data will not be dropped for example if a MS has field_name=['field_0','field_10','field_08'] and ps.sel(field_name='field_0') is done the resulting MS will still have field_name=['field_0','field_10','field_08'].

        Examples:
            ps.sel(intents='OBSERVE_TARGET#ON_SOURCE', polarization=['RR', 'LL']) # Select all MSs with intents 'OBSERVE_TARGET#ON_SOURCE' and polarization 'RR' or 'LL'.
            ps.sel(query='start_frequency > 100e9 AND end_frequency < 200e9') # Select all MSs with start_frequency greater than 100 GHz and less than 200 GHz.

        Args:
            query (str): A Pandas query string. Default is None.
            string_exact_match (bool): If True, the selection will be an exact match for string and string list columns. Default is True.
            **kwargs: Keyword arguments representing column names and values to filter the Processing Set.

        Returns:
            processing_set: The subset of the Processing Set.
        """
        import numpy as np

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

        sub_ps = ProcessingSet()
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
        sub_ps = ProcessingSet()
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
        sub_ps = ProcessingSet()
        for key, val in self.items():
            sub_ps[key] = val.isel(kwargs)
        return sub_ps

    def to_store(self, store, **kwargs):
        """
        Write the Processing Set to a Zarr store.
        Does not write to cloud storage yet.

        Args:
            store (str): The path to the Zarr store.
            **kwargs: Additional keyword arguments to be passed to `xarray.Dataset.to_zarr`. See https://docs.xarray.dev/en/latest/generated/xarray.Dataset.to_zarr.html for more information.

        Returns:
            None
        """
        import os

        for key, value in self.items():
            value.to_store(os.path.join(store, key), **kwargs)

    def get_combined_field_and_source_xds(self, data_group="base"):
        """
        Returns an xarray.Dataset combining the field_and_source_xds of all cor_xds's in a Processing Set for a given data_group.
        The combined xarray.Dataset will have a new dimension 'correlated_xds_name' which will be the name of the cor_xds.

        Returns:
            list: A list containing the phase center coordinates.
        """

        df = self.summary(data_group)

        # if "Ephemeris" in list(df['field_coords']):
        #     logger.warning("Cannot combine ephemeris field_and_source_xds yet.")
        #     return None

        combined_field_and_source_xds = xr.Dataset()
        combined_ephemeris_field_and_source_xds = xr.Dataset()
        for cor_name, cor_xds in self.items():
            correlated_data_name = cor_xds.attrs["data_groups"][data_group][
                "correlated_data"
            ]

            field_and_source_xds = (
                cor_xds[correlated_data_name]
                .attrs["field_and_source_xds"]
                .copy(deep=True)
            )

            if (
                "line_name" in field_and_source_xds.coords
            ):  # Not including line info since it is a function of spw.
                field_and_source_xds = field_and_source_xds.drop_vars(
                    ["LINE_REST_FREQUENCY", "LINE_SYSTEMIC_VELOCITY"], errors="ignore"
                )
                del field_and_source_xds["line_name"]
                del field_and_source_xds["line_label"]

            if "time" in field_and_source_xds.coords:
                if "time" not in field_and_source_xds.field_name.dims:
                    field_names = np.array(
                        [field_and_source_xds.field_name.values.item()]
                        * len(field_and_source_xds.time.values)
                    )
                    source_names = np.array(
                        [field_and_source_xds.source_name.values.item()]
                        * len(field_and_source_xds.time.values)
                    )
                    del field_and_source_xds["field_name"]
                    del field_and_source_xds["source_name"]
                    field_and_source_xds = field_and_source_xds.assign_coords(
                        field_name=("time", field_names)
                    )
                    field_and_source_xds = field_and_source_xds.assign_coords(
                        source_name=("time", source_names)
                    )
                field_and_source_xds = field_and_source_xds.swap_dims(
                    {"time": "field_name"}
                )
                del field_and_source_xds["time"]
            elif "time_ephemeris" in field_and_source_xds.coords:
                if "time_ephemeris" not in field_and_source_xds.field_name.dims:
                    field_names = np.array(
                        [field_and_source_xds.field_name.values.item()]
                        * len(field_and_source_xds.time.values)
                    )
                    source_names = np.array(
                        [field_and_source_xds.source_name.values.item()]
                        * len(field_and_source_xds.time.values)
                    )
                    del field_and_source_xds["field_name"]
                    del field_and_source_xds["source_name"]
                    field_and_source_xds = field_and_source_xds.assign_coords(
                        field_name=("time_ephemeris", field_names)
                    )
                    field_and_source_xds = field_and_source_xds.assign_coords(
                        source_name=("time_ephemeris", source_names)
                    )
                field_and_source_xds = field_and_source_xds.swap_dims(
                    {"time_ephemeris": "field_name"}
                )
                del field_and_source_xds["time_ephemeris"]
            else:
                for dv_names in field_and_source_xds.data_vars:
                    if "field_name" not in field_and_source_xds[dv_names].dims:
                        field_and_source_xds[dv_names] = field_and_source_xds[
                            dv_names
                        ].expand_dims("field_name")

            if field_and_source_xds.is_ephemeris:
                if len(combined_ephemeris_field_and_source_xds.data_vars) == 0:
                    combined_ephemeris_field_and_source_xds = field_and_source_xds
                else:
                    combined_ephemeris_field_and_source_xds = xr.concat(
                        [combined_ephemeris_field_and_source_xds, field_and_source_xds],
                        dim="field_name",
                    )
            else:
                if len(combined_field_and_source_xds.data_vars) == 0:
                    combined_field_and_source_xds = field_and_source_xds
                else:
                    combined_field_and_source_xds = xr.concat(
                        [combined_field_and_source_xds, field_and_source_xds],
                        dim="field_name",
                    )

        if (len(combined_field_and_source_xds.data_vars) > 0) and (
            "FIELD_PHASE_CENTER" in combined_field_and_source_xds
        ):
            combined_field_and_source_xds = (
                combined_field_and_source_xds.drop_duplicates("field_name")
            )

            combined_field_and_source_xds["MEAN_PHASE_CENTER"] = (
                combined_field_and_source_xds["FIELD_PHASE_CENTER"].mean(
                    dim=["field_name"]
                )
            )

            ra1 = (
                combined_field_and_source_xds["FIELD_PHASE_CENTER"]
                .sel(sky_dir_label="ra")
                .values
            )
            dec1 = (
                combined_field_and_source_xds["FIELD_PHASE_CENTER"]
                .sel(sky_dir_label="dec")
                .values
            )
            ra2 = (
                combined_field_and_source_xds["MEAN_PHASE_CENTER"]
                .sel(sky_dir_label="ra")
                .values
            )
            dec2 = (
                combined_field_and_source_xds["MEAN_PHASE_CENTER"]
                .sel(sky_dir_label="dec")
                .values
            )

            from xradio._utils.coord_math import haversine

            distance = haversine(ra1, dec1, ra2, dec2)
            min_index = distance.argmin()

            combined_field_and_source_xds.attrs["center_field_name"] = (
                combined_field_and_source_xds.field_name[min_index].values
            )

        if (len(combined_ephemeris_field_and_source_xds.data_vars) > 0) and (
            "FIELD_PHASE_CENTER" in combined_ephemeris_field_and_source_xds
        ):
            combined_ephemeris_field_and_source_xds = (
                combined_ephemeris_field_and_source_xds.drop_duplicates("field_name")
            )

            from xradio._utils.coord_math import wrap_to_pi

            offset = (
                combined_ephemeris_field_and_source_xds["FIELD_PHASE_CENTER"]
                - combined_ephemeris_field_and_source_xds["SOURCE_LOCATION"]
            )
            combined_ephemeris_field_and_source_xds["FIELD_OFFSET"] = xr.DataArray(
                wrap_to_pi(offset.sel(sky_pos_label=["ra", "dec"])).values,
                dims=["field_name", "sky_dir_label"],
            )
            combined_ephemeris_field_and_source_xds["FIELD_OFFSET"].attrs = (
                combined_ephemeris_field_and_source_xds["FIELD_PHASE_CENTER"].attrs
            )
            combined_ephemeris_field_and_source_xds["FIELD_OFFSET"].attrs["units"] = (
                combined_ephemeris_field_and_source_xds["FIELD_OFFSET"].attrs["units"][
                    :2
                ]
            )

            ra1 = (
                combined_ephemeris_field_and_source_xds["FIELD_OFFSET"]
                .sel(sky_dir_label="ra")
                .values
            )
            dec1 = (
                combined_ephemeris_field_and_source_xds["FIELD_OFFSET"]
                .sel(sky_dir_label="dec")
                .values
            )
            ra2 = 0.0
            dec2 = 0.0

            from xradio._utils.coord_math import haversine

            distance = haversine(ra1, dec1, ra2, dec2)
            min_index = distance.argmin()

            combined_ephemeris_field_and_source_xds.attrs["center_field_name"] = (
                combined_ephemeris_field_and_source_xds.field_name[min_index].values
            )

        return combined_field_and_source_xds, combined_ephemeris_field_and_source_xds

    def plot_phase_centers(self, label_all_fields=False, data_group="base"):
        """
        Used for Mosaics. Plots the phase center locations of all fields in the Processing Set.

        Parameters
        ----------
        data_group : _type_
            _description_
        """
        combined_field_and_source_xds, combined_ephemeris_field_and_source_xds = (
            self.get_combined_field_and_source_xds(data_group)
        )
        from matplotlib import pyplot as plt

        if len(combined_field_and_source_xds.data_vars) > 0:
            plt.figure()
            plt.title("Field Phase Center Locations")
            plt.scatter(
                combined_field_and_source_xds["FIELD_PHASE_CENTER"].sel(
                    sky_dir_label="ra"
                ),
                combined_field_and_source_xds["FIELD_PHASE_CENTER"].sel(
                    sky_dir_label="dec"
                ),
            )

            center_field_name = combined_field_and_source_xds.attrs["center_field_name"]
            center_field = combined_field_and_source_xds.sel(
                field_name=center_field_name
            )
            plt.scatter(
                center_field["FIELD_PHASE_CENTER"].sel(sky_dir_label="ra"),
                center_field["FIELD_PHASE_CENTER"].sel(sky_dir_label="dec"),
                color="red",
                label=center_field_name,
            )
            plt.xlabel("RA (rad)")
            plt.ylabel("DEC (rad)")
            plt.legend()
            plt.show()

        if len(combined_ephemeris_field_and_source_xds.data_vars) > 0:

            plt.figure()
            plt.title(
                "Offset of Field Phase Center from Source Location (Ephemeris Data)"
            )
            plt.scatter(
                combined_ephemeris_field_and_source_xds["FIELD_OFFSET"].sel(
                    sky_dir_label="ra"
                ),
                combined_ephemeris_field_and_source_xds["FIELD_OFFSET"].sel(
                    sky_dir_label="dec"
                ),
            )

            center_field_name = combined_ephemeris_field_and_source_xds.attrs[
                "center_field_name"
            ]
            center_field = combined_ephemeris_field_and_source_xds.sel(
                field_name=center_field_name
            )
            plt.scatter(
                center_field["FIELD_OFFSET"].sel(sky_dir_label="ra"),
                center_field["FIELD_OFFSET"].sel(sky_dir_label="dec"),
                color="red",
                label=center_field_name,
            )
            plt.xlabel("RA Offset (rad)")
            plt.ylabel("DEC Offset (rad)")
            plt.legend()
            plt.show()

    def get_combined_antenna_xds(self):
        """ """
        combined_antenna_xds = xr.Dataset()
        for cor_name, cor_xds in self.items():
            antenna_xds = cor_xds.antenna_xds.copy(deep=True)

            if len(combined_antenna_xds.data_vars) == 0:
                combined_antenna_xds = antenna_xds
            else:
                combined_antenna_xds = xr.concat(
                    [combined_antenna_xds, antenna_xds],
                    dim="antenna_name",
                    data_vars="minimal",
                    coords="minimal",
                )

        # ALMA WVR antenna_xds data has a NaN value for the antenna receptor angle.
        if "ANTENNA_RECEPTOR_ANGLE" in combined_antenna_xds.data_vars:
            combined_antenna_xds = combined_antenna_xds.dropna("antenna_name")

        combined_antenna_xds = combined_antenna_xds.drop_duplicates("antenna_name")

        return combined_antenna_xds

    def plot_antenna_positions(self):
        """
        Plots the antenna positions of all antennas in the Processing Set.
        """
        combined_antenna_xds = self.get_combined_antenna_xds()
        from matplotlib import pyplot as plt

        plt.figure()
        plt.title("Antenna Positions")
        plt.scatter(
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="x"),
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="y"),
        )
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()

        plt.figure()
        plt.title("Antenna Positions")
        plt.scatter(
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="x"),
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="z"),
        )
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.show()

        plt.figure()
        plt.title("Antenna Positions")
        plt.scatter(
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="y"),
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="z"),
        )
        plt.xlabel("y (m)")
        plt.ylabel("z (m)")
        plt.show()
