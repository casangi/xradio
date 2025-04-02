import pandas as pd
from xradio._utils.list_and_array import to_list
import numbers
import numpy as np
import toolviper.utils.logger as logger
import xarray as xr

PS_DATASET_TYPES = {"processing_set"}


class InvalidAccessorLocation(ValueError):
    """
    Raised by Processing Set accessor functions called on a wrong DataTree node (not processing set).
    """

    pass


@xr.register_datatree_accessor("xr_ps")
class ProcessingSetXdt:
    """
    Accessor to Processing Set DataTree nodes. Provides Processing Set specific functionality such
    as producing a summary of the processing set (with information from all its MSv4s), or retrieving
    combined antenna or field_and_source datasets.
    """

    _xdt: xr.DataTree

    def __init__(self, datatree: xr.DataTree):
        """
        Initialize the ProcessingSetXdt instance.

        Parameters
        ----------
        datatree: xarray.DataTree
            The Processing Set DataTree node to construct a ProcessingSetXdt accessor.
        """

        self._xdt = datatree
        self.meta = {"summary": {}}

    def summary(self, data_group: str = "base") -> pd.DataFrame:
        """
        Generate and retrieve a summary of the Processing Set.

        The summary includes information such as the names of the Measurement Sets,
        their intents, polarizations, spectral window names, field names, source names,
        field coordinates, start frequencies, and end frequencies.

        Parameters
        ----------
        data_group : str, optional
            The data group to summarize. Default is "base".

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the summary information of the specified data group.
        """

        if self._xdt.attrs.get("type") not in PS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a processing set node."
            )

        if data_group in self.meta["summary"]:
            return self.meta["summary"][data_group]
        else:
            self.meta["summary"][data_group] = self._summary(data_group).sort_values(
                by=["name"], ascending=True
            )
            return self.meta["summary"][data_group]

    def get_max_dims(self) -> dict[str, int]:
        """
        Determine the maximum dimensions across all Measurement Sets in the Processing Set.

        This method examines each Measurement Set's dimensions and computes the maximum
        size for each dimension across the entire Processing Set.

        For example, if the Processing Set contains two MSs with dimensions (50, 20, 30) and (10, 30, 40),
        the maximum dimensions will be (50, 30, 40).

        Returns
        -------
        dict
            A dictionary containing the maximum dimensions of the Processing Set, with dimension names as keys
            and their maximum sizes as values.
        """

        if self._xdt.attrs.get("type") not in PS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a processing set node."
            )

        if "max_dims" in self.meta:
            return self.meta["max_dims"]
        else:
            max_dims = None
            for ms_xdt in self._xdt.values():
                if max_dims is None:
                    max_dims = dict(ms_xdt.sizes)
                else:
                    for dim_name, size in ms_xdt.sizes.items():
                        if dim_name in max_dims:
                            if max_dims[dim_name] < size:
                                max_dims[dim_name] = size
                        else:
                            max_dims[dim_name] = size
            self.meta["max_dims"] = max_dims
            return self.meta["max_dims"]

    def get_freq_axis(self) -> xr.DataArray:
        """
        Combine the frequency axes of all Measurement Sets in the Processing Set.

        This method aggregates the frequency information from each Measurement Set to create
        a unified frequency axis for the entire Processing Set.

        Returns
        -------
        xarray.DataArray
            The combined frequency axis of the Processing Set.
        """
        if self._xdt.attrs.get("type") not in PS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a processing set node."
            )

        if "freq_axis" in self.meta:
            return self.meta["freq_axis"]
        else:
            spw_ids = []
            freq_axis_list = []
            frame = self._xdt[next(iter(self._xdt.children))].frequency.attrs[
                "observer"
            ]
            for ms_xdt in self._xdt.values():
                assert (
                    frame == ms_xdt.frequency.attrs["observer"]
                ), "Frequency reference frame not consistent in Processing Set."
                if ms_xdt.frequency.attrs["spectral_window_id"] not in spw_ids:
                    spw_ids.append(ms_xdt.frequency.attrs["spectral_window_id"])
                    freq_axis_list.append(ms_xdt.frequency)

            freq_axis = xr.concat(freq_axis_list, dim="frequency").sortby("frequency")
            self.meta["freq_axis"] = freq_axis
            return self.meta["freq_axis"]

    def _summary(self, data_group: str = "base"):
        summary_data = {
            "name": [],
            "intents": [],
            "shape": [],
            "polarization": [],
            "scan_name": [],
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

        for key, value in self._xdt.items():
            partition_info = value.xr_ms.get_partition_info()

            summary_data["name"].append(key)
            summary_data["intents"].append(partition_info["intents"])
            summary_data["spw_name"].append(partition_info["spectral_window_name"])
            summary_data["polarization"].append(value.polarization.values)
            summary_data["scan_name"].append(partition_info["scan_name"])
            data_name = value.attrs["data_groups"][data_group]["correlated_data"]

            if "VISIBILITY" in data_name:
                center_name = "FIELD_PHASE_CENTER"

            if "SPECTRUM" in data_name:
                center_name = "FIELD_REFERENCE_CENTER"

            summary_data["shape"].append(value[data_name].shape)

            summary_data["field_name"].append(partition_info["field_name"])
            summary_data["source_name"].append(partition_info["source_name"])

            summary_data["line_name"].append(partition_info["line_name"])

            summary_data["start_frequency"].append(
                to_list(value["frequency"].values)[0]
            )
            summary_data["end_frequency"].append(to_list(value["frequency"].values)[-1])

            field_and_source_xds = value["field_and_source_xds_" + data_group]

            if field_and_source_xds.attrs["type"] == "field_and_source_ephemeris":
                summary_data["field_coords"].append("Ephemeris")
            elif field_and_source_xds[center_name]["field_name"].size > 1:
                summary_data["field_coords"].append("Multi-Phase-Center")
            else:
                ra_dec_rad = field_and_source_xds[center_name].values[0, :]
                frame = field_and_source_xds[center_name].attrs["frame"].lower()

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

    def query(
        self, string_exact_match: bool = True, query: str = None, **kwargs
    ) -> xr.DataTree:
        """
        Select a subset of the Processing Set based on specified criteria.

        This method allows filtering the Processing Set by matching column names and values
        or by applying a Pandas query string. The selection criteria can target various
        attributes of the Measurement Sets such as intents, polarization, spectral window names, etc.

        A data group can be selected by name by using the `data_group_name` parameter. This is applied to each Measurement Set in the Processing Set.

        Note
        ----
        This selection does not modify the actual data within the Measurement Sets. For example, if
        a Measurement Set has `field_name=['field_0','field_10','field_08']` and `ps.query(field_name='field_0')`
        is invoked, the resulting subset will still contain the original list `['field_0','field_10','field_08']`.
        The exception is data group selection, using `data_group_name`, that will select data variables only associated with the specified data group in the Measurement Set.

        Parameters
        ----------
        string_exact_match : bool, optional
            If `True`, string matching will require exact matches for string and string list columns.
            If `False`, partial matches are allowed. Default is `True`.
        query : str, optional
            A Pandas query string to apply additional filtering. Default is `None`.
        **kwargs : dict
            Keyword arguments representing column names and their corresponding values to filter the Processing Set.

        Returns
        -------
        xr.DataTree
            A new Processing Set DataTree instance containing only the Measurement Sets that match the selection criteria.

        Examples
        --------
        >>> # Select all MSs with intents 'OBSERVE_TARGET#ON_SOURCE' and polarization 'RR' or 'LL'
        >>> selected_ps = ps.query(intents='OBSERVE_TARGET#ON_SOURCE', polarization=['RR', 'LL'])

        >>> # Select all MSs with start_frequency greater than 100 GHz and less than 200 GHz
        >>> selected_ps = ps.query(query='start_frequency > 100e9 AND end_frequency < 200e9')
        """

        if self._xdt.attrs.get("type") not in PS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a processing set node."
            )

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
        data_group_name = None
        for key, value in kwargs.items():

            if "data_group_name" == key:
                data_group_name = value
            else:
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

        sub_ps_xdt = xr.DataTree()
        for key, val in self._xdt.items():
            if key in summary_table["name"].values:
                if data_group_name is not None:
                    sub_ps_xdt[key] = val.xr_ms.sel(data_group_name=data_group_name)
                else:
                    sub_ps_xdt[key] = val

        sub_ps_xdt.attrs = self._xdt.attrs

        return sub_ps_xdt

    def get_combined_field_and_source_xds(self, data_group: str = "base") -> xr.Dataset:
        """
        Combine all non-ephemeris `field_and_source_xds` datasets from a Processing Set for a data group into a
        single dataset.

        Parameters
        ----------
        data_group : str, optional
            The data group to process. Default is "base".

        Returns
        -------
        xarray.Dataset
            combined_field_and_source_xds: Combined dataset for standard (non-ephemeris) fields.

        Raises
        ------
        ValueError
            If the `field_and_source_xds` attribute is missing or improperly formatted in any Measurement Set.
        """

        if self._xdt.attrs.get("type") not in PS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a processing set node."
            )

        combined_field_and_source_xds = xr.Dataset()
        for ms_name, ms_xdt in self._xdt.items():
            correlated_data_name = ms_xdt.attrs["data_groups"][data_group][
                "correlated_data"
            ]

            field_and_source_xds = ms_xdt["field_and_source_xds_" + data_group].ds

            if not field_and_source_xds.attrs["type"] == "field_and_source_ephemeris":

                if (
                    "line_name" in field_and_source_xds.coords
                ):  # Not including line info since it is a function of spw.
                    field_and_source_xds = field_and_source_xds.drop_vars(
                        ["LINE_REST_FREQUENCY", "LINE_SYSTEMIC_VELOCITY"],
                        errors="ignore",
                    )
                    del field_and_source_xds["line_name"]
                    del field_and_source_xds["line_label"]

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

        return combined_field_and_source_xds

    def get_combined_field_and_source_xds_ephemeris(
        self, data_group: str = "base"
    ) -> xr.Dataset:
        """
        Combine all ephemeris `field_and_source_xds` datasets from a Processing Set for a datagroup into a single dataset.

        Parameters
        ----------
        data_group : str, optional
            The data group to process. Default is "base".

        Returns
        -------
        xarray.Dataset
            combined_ephemeris_field_and_source_xds: Combined dataset for ephemeris fields.

        Raises
        ------
        ValueError
            If the `field_and_source_xds` attribute is missing or improperly formatted in any Measurement Set.
        """

        if self._xdt.attrs.get("type") not in PS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a processing set node."
            )

        combined_ephemeris_field_and_source_xds = xr.Dataset()
        for ms_name, ms_xdt in self._xdt.items():

            correlated_data_name = ms_xdt.attrs["data_groups"][data_group][
                "correlated_data"
            ]

            field_and_source_xds = field_and_source_xds = ms_xdt[
                "field_and_source_xds_" + data_group
            ].ds

            if field_and_source_xds.attrs["type"] == "field_and_source_ephemeris":

                if (
                    "line_name" in field_and_source_xds.coords
                ):  # Not including line info since it is a function of spw.
                    field_and_source_xds = field_and_source_xds.drop_vars(
                        ["LINE_REST_FREQUENCY", "LINE_SYSTEMIC_VELOCITY"],
                        errors="ignore",
                    )
                    del field_and_source_xds["line_name"]
                    del field_and_source_xds["line_label"]

                from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
                    interpolate_to_time,
                )

                if "time_ephemeris" in field_and_source_xds:
                    field_and_source_xds = interpolate_to_time(
                        field_and_source_xds,
                        field_and_source_xds.time,
                        "field_and_source_xds",
                        "time_ephemeris",
                    )
                    del field_and_source_xds["time_ephemeris"]
                    field_and_source_xds = field_and_source_xds.rename(
                        {"time_ephemeris": "time"}
                    )

                if "OBSERVER_POSITION" in field_and_source_xds:
                    field_and_source_xds = field_and_source_xds.drop_vars(
                        ["OBSERVER_POSITION"], errors="ignore"
                    )

                if len(combined_ephemeris_field_and_source_xds.data_vars) == 0:
                    combined_ephemeris_field_and_source_xds = field_and_source_xds
                else:

                    combined_ephemeris_field_and_source_xds = xr.concat(
                        [combined_ephemeris_field_and_source_xds, field_and_source_xds],
                        dim="time",
                    )

        if (len(combined_ephemeris_field_and_source_xds.data_vars) > 0) and (
            "FIELD_PHASE_CENTER" in combined_ephemeris_field_and_source_xds
        ):

            from xradio._utils.coord_math import wrap_to_pi

            offset = (
                combined_ephemeris_field_and_source_xds["FIELD_PHASE_CENTER"]
                - combined_ephemeris_field_and_source_xds["SOURCE_LOCATION"]
            )
            combined_ephemeris_field_and_source_xds["FIELD_OFFSET"] = xr.DataArray(
                wrap_to_pi(offset.sel(sky_pos_label=["ra", "dec"])).values,
                dims=["time", "sky_dir_label"],
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

        return combined_ephemeris_field_and_source_xds

    def plot_phase_centers(
        self, label_all_fields: bool = False, data_group: str = "base"
    ):
        """
        Plot the phase center locations of all fields in the Processing Set.

        This method is primarily used for visualizing mosaics. It generates scatter plots of
        the phase center coordinates for both standard and ephemeris fields. The central field
        is highlighted in red based on the closest phase center calculation.

        Parameters
        ----------
        label_all_fields : bool, optional
            If `True`, all fields will be labeled on the plot. Default is `False`.
        data_group : str, optional
            The data group to use for processing. Default is "base".

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the combined datasets are empty or improperly formatted.
        """
        if self._xdt.attrs.get("type") not in PS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a processing set node."
            )

        combined_field_and_source_xds = self.get_combined_field_and_source_xds(
            data_group
        )
        combined_ephemeris_field_and_source_xds = (
            self.get_combined_field_and_source_xds_ephemeris(data_group)
        )
        from matplotlib import pyplot as plt

        if (len(combined_field_and_source_xds.data_vars) > 0) and (
            "FIELD_PHASE_CENTER" in combined_field_and_source_xds
        ):
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

        if (len(combined_ephemeris_field_and_source_xds.data_vars) > 0) and (
            "FIELD_PHASE_CENTER" in combined_ephemeris_field_and_source_xds
        ):

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

            combined_ephemeris_field_and_source_xds = (
                combined_ephemeris_field_and_source_xds.set_xindex("field_name")
            )

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

    def get_combined_antenna_xds(self) -> xr.Dataset:
        """
        Combine the `antenna_xds` datasets from all Measurement Sets into a single dataset.

        This method concatenates the antenna datasets from each Measurement Set along the 'antenna_name' dimension.

        Returns
        -------
        xarray.Dataset
            A combined `xarray.Dataset` containing antenna information from all Measurement Sets.

        Raises
        ------
        ValueError
            If antenna datasets are missing required variables or improperly formatted.
        """
        if self._xdt.attrs.get("type") not in PS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a processing set node."
            )

        combined_antenna_xds = xr.Dataset()
        for cor_name, ms_xdt in self._xdt.items():
            antenna_xds = ms_xdt.antenna_xds.ds

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
        Plot the antenna positions of all antennas in the Processing Set.

        This method generates three scatter plots displaying the antenna positions in different planes:
        - X vs Y
        - X vs Z
        - Y vs Z

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the combined antenna dataset is empty or missing required coordinates.
        """
        if self._xdt.attrs.get("type") not in PS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a processing set node."
            )

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


# class ProcessingSet(dict):
#     """
#     A dictionary subclass representing a Processing Set (PS) containing Measurement Sets v4 (MS).

#     This class extends the built-in `dict` class to provide additional methods for
#     manipulating and selecting subsets of the Processing Set. It includes functionality
#     for summarizing metadata, selecting subsets based on various criteria, and
#     exporting the data to storage formats.

#     Parameters
#     ----------
#     *args : dict, optional
#         Variable length argument list passed to the base `dict` class.
#     **kwargs : dict, optional
#         Arbitrary keyword arguments passed to the base `dict` class.
#     """

#     def __init__(self, *args, **kwargs):
#         """
#         Initialize the ProcessingSet instance.

#         Parameters
#         ----------
#         *args : dict, optional
#             Variable length argument list passed to the base `dict` class.
#         **kwargs : dict, optional
#             Arbitrary keyword arguments passed to the base `dict` class.
#         """
#         super().__init__(*args, **kwargs)
#         self.meta = {"summary": {}}

#     def summary(self, data_group="base"):
#         """
#         Generate and retrieve a summary of the Processing Set.

#         The summary includes information such as the names of the Measurement Sets,
#         their intents, polarizations, spectral window names, field names, source names,
#         field coordinates, start frequencies, and end frequencies.

#         Parameters
#         ----------
#         data_group : str, optional
#             The data group to summarize. Default is "base".

#         Returns
#         -------
#         pandas.DataFrame
#             A DataFrame containing the summary information of the specified data group.
#         """

#         if data_group in self.meta["summary"]:
#             return self.meta["summary"][data_group]
#         else:
#             self.meta["summary"][data_group] = self._summary(data_group).sort_values(
#                 by=["name"], ascending=True
#             )
#             return self.meta["summary"][data_group]

#     def get_ps_max_dims(self):
#         """
#         Determine the maximum dimensions across all Measurement Sets in the Processing Set.

#         This method examines each Measurement Set's dimensions and computes the maximum
#         size for each dimension across the entire Processing Set.

#         For example, if the Processing Set contains two MSs with dimensions (50, 20, 30) and (10, 30, 40),
#         the maximum dimensions will be (50, 30, 40).

#         Returns
#         -------
#         dict
#             A dictionary containing the maximum dimensions of the Processing Set, with dimension names as keys
#             and their maximum sizes as values.
#         """
#         if "max_dims" in self.meta:
#             return self.meta["max_dims"]
#         else:
#             self.meta["max_dims"] = self._get_ps_max_dims()
#             return self.meta["max_dims"]

#     def get_ps_freq_axis(self):
#         """
#         Combine the frequency axes of all Measurement Sets in the Processing Set.

#         This method aggregates the frequency information from each Measurement Set to create
#         a unified frequency axis for the entire Processing Set.

#         Returns
#         -------
#         xarray.DataArray
#             The combined frequency axis of the Processing Set.
#         """
#         if "freq_axis" in self.meta:
#             return self.meta["freq_axis"]
#         else:
#             self.meta["freq_axis"] = self._get_ps_freq_axis()
#             return self.meta["freq_axis"]

#     def _summary(self, data_group="base"):
#         summary_data = {
#             "name": [],
#             "intents": [],
#             "shape": [],
#             "polarization": [],
#             "scan_name": [],
#             "spw_name": [],
#             "field_name": [],
#             "source_name": [],
#             "line_name": [],
#             "field_coords": [],
#             "start_frequency": [],
#             "end_frequency": [],
#         }
#         from astropy.coordinates import SkyCoord
#         import astropy.units as u

#         for key, value in self.items():
#             summary_data["name"].append(key)
#             summary_data["intents"].append(partition_info["intents"])
#             summary_data["spw_name"].append(
#                 partition_info["spectral_window_name"]
#             )
#             summary_data["polarization"].append(value.polarization.values)
#             summary_data["scan_name"].append(partition_info["scan_name"])
#             data_name = value.attrs["data_groups"][data_group]["correlated_data"]

#             if "VISIBILITY" in data_name:
#                 center_name = "FIELD_PHASE_CENTER"

#             if "SPECTRUM" in data_name:
#                 center_name = "FIELD_REFERENCE_CENTER"

#             summary_data["shape"].append(value[data_name].shape)

#             summary_data["field_name"].append(
#                 partition_info["field_name"]
#             )
#             summary_data["source_name"].append(
#                 partition_info["source_name"]
#             )

#             summary_data["line_name"].append(partition_info["line_name"])

#             summary_data["start_frequency"].append(
#                 to_list(value["frequency"].values)[0]
#             )
#             summary_data["end_frequency"].append(to_list(value["frequency"].values)[-1])

#             if (
#                 value[data_name].attrs["field_and_source_xds"].attrs["type"]
#                 == "field_and_source_ephemeris"
#             ):
#                 summary_data["field_coords"].append("Ephemeris")
#             # elif (
#             #     "time"
#             #     in value[data_name].attrs["field_and_source_xds"][center_name].coords
#             # ):
#             elif (
#                 value[data_name]
#                 .attrs["field_and_source_xds"][center_name]["field_name"]
#                 .size
#                 > 1
#             ):
#                 summary_data["field_coords"].append("Multi-Phase-Center")
#             else:
#                 ra_dec_rad = (
#                     value[data_name]
#                     .attrs["field_and_source_xds"][center_name]
#                     .values[0, :]
#                 )
#                 frame = (
#                     value[data_name]
#                     .attrs["field_and_source_xds"][center_name]
#                     .attrs["frame"]
#                     .lower()
#                 )

#                 coord = SkyCoord(
#                     ra=ra_dec_rad[0] * u.rad, dec=ra_dec_rad[1] * u.rad, frame=frame
#                 )

#                 summary_data["field_coords"].append(
#                     [
#                         frame,
#                         coord.ra.to_string(unit=u.hour, precision=2),
#                         coord.dec.to_string(unit=u.deg, precision=2),
#                     ]
#                 )

#         summary_df = pd.DataFrame(summary_data)
#         return summary_df

#     def _get_ps_freq_axis(self):

#         spw_ids = []
#         freq_axis_list = []
#         frame = self.get(0).frequency.attrs["observer"]
#         for ms_xds in self.values():
#             assert (
#                 frame == ms_xds.frequency.attrs["observer"]
#             ), "Frequency reference frame not consistent in Processing Set."
#             if ms_xds.frequency.attrs["spectral_window_id"] not in spw_ids:
#                 spw_ids.append(ms_xds.frequency.attrs["spectral_window_id"])
#                 freq_axis_list.append(ms_xds.frequency)

#         freq_axis = xr.concat(freq_axis_list, dim="frequency").sortby("frequency")
#         return freq_axis

#     def _get_ps_max_dims(self):
#         max_dims = None
#         for ms_xds in self.values():
#             if max_dims is None:
#                 max_dims = dict(ms_xds.sizes)
#             else:
#                 for dim_name, size in ms_xds.sizes.items():
#                     if dim_name in max_dims:
#                         if max_dims[dim_name] < size:
#                             max_dims[dim_name] = size
#                     else:
#                         max_dims[dim_name] = size
#         return max_dims

#     def get(self, id):
#         return self[list(self.keys())[id]]

#     def sel(self, string_exact_match: bool = True, query: str = None, **kwargs):
#         """
#         Select a subset of the Processing Set based on specified criteria.

#         This method allows filtering the Processing Set by matching column names and values
#         or by applying a Pandas query string. The selection criteria can target various
#         attributes of the Measurement Sets such as intents, polarization, spectral window names, etc.

#         Note
#         ----
#         This selection does not modify the actual data within the Measurement Sets. For example, if
#         a Measurement Set has `field_name=['field_0','field_10','field_08']` and `ps.query(field_name='field_0')`
#         is invoked, the resulting subset will still contain the original list `['field_0','field_10','field_08']`.

#         Parameters
#         ----------
#         string_exact_match : bool, optional
#             If `True`, string matching will require exact matches for string and string list columns.
#             If `False`, partial matches are allowed. Default is `True`.
#         query : str, optional
#             A Pandas query string to apply additional filtering. Default is `None`.
#         **kwargs : dict
#             Keyword arguments representing column names and their corresponding values to filter the Processing Set.

#         Returns
#         -------
#         ProcessingSet
#             A new `ProcessingSet` instance containing only the Measurement Sets that match the selection criteria.

#         Examples
#         --------
#         >>> # Select all MSs with intents 'OBSERVE_TARGET#ON_SOURCE' and polarization 'RR' or 'LL'
#         >>> selected_ps = ps.query(intents='OBSERVE_TARGET#ON_SOURCE', polarization=['RR', 'LL'])

#         >>> # Select all MSs with start_frequency greater than 100 GHz and less than 200 GHz
#         >>> selected_ps = ps.query(query='start_frequency > 100e9 AND end_frequency < 200e9')
#         """
#         import numpy as np

#         def select_rows(df, col, sel_vals, string_exact_match):
#             def check_selection(row_val):
#                 row_val = to_list(
#                     row_val
#                 )  # make sure that it is a list so that we can iterate over it.

#                 for rw in row_val:
#                     for s in sel_vals:
#                         if string_exact_match:
#                             if rw == s:
#                                 return True
#                         else:
#                             if s in rw:
#                                 return True
#                 return False

#             return df[df[col].apply(check_selection)]

#         summary_table = self.summary()
#         for key, value in kwargs.items():
#             value = to_list(value)  # make sure value is a list.

#             if len(value) == 1 and isinstance(value[0], slice):
#                 summary_table = summary_table[
#                     summary_table[key].between(value[0].start, value[0].stop)
#                 ]
#             else:
#                 summary_table = select_rows(
#                     summary_table, key, value, string_exact_match
#                 )

#         if query is not None:
#             summary_table = summary_table.query(query)

#         sub_ps = ProcessingSet()
#         for key, val in self.items():
#             if key in summary_table["name"].values:
#                 sub_ps[key] = val

#         return sub_ps

#     def ms_sel(self, **kwargs):
#         """
#         Select a subset of the Processing Set by applying the `xarray.Dataset.sel` method to each Measurement Set.

#         This method allows for selection based on label-based indexing for each dimension of the datasets.

#         Parameters
#         ----------
#         **kwargs : dict
#             Keyword arguments representing dimension names and the labels to select along those dimensions.
#             These are passed directly to the `xarray.Dataset.sel <https://docs.xarray.dev/en/latest/generated/xarray.Dataset.sel.html>`__ method.

#         Returns
#         -------
#         ProcessingSet
#             A new `ProcessingSet` instance containing the selected subsets of each Measurement Set.
#         """
#         sub_ps = ProcessingSet()
#         for key, val in self.items():
#             sub_ps[key] = val.sel(kwargs)
#         return sub_ps

#     def ms_isel(self, **kwargs):
#         """
#         Select a subset of the Processing Set by applying the `isel` method to each Measurement Set.

#         This method allows for selection based on integer-based indexing for each dimension of the datasets.

#         Parameters
#         ----------
#         **kwargs : dict
#             Keyword arguments representing dimension names and the integer indices to select along those dimensions.
#             These are passed directly to the `xarray.Dataset.isel <https://docs.xarray.dev/en/latest/generated/xarray.Dataset.isel.html>`__ method.

#         Returns
#         -------
#         ProcessingSet
#             A new `ProcessingSet` instance containing the selected subsets of each Measurement Set.
#         """
#         sub_ps = ProcessingSet()
#         for key, val in self.items():
#             sub_ps[key] = val.isel(kwargs)
#         return sub_ps

#     def to_store(self, store, **kwargs):
#         """
#         Write the Processing Set to a Zarr store.

#         This method serializes each Measurement Set within the Processing Set to a separate Zarr group
#         within the specified store directory. Note that writing to cloud storage is not supported yet.

#         Parameters
#         ----------
#         store : str
#             The filesystem path to the Zarr store directory where the data will be saved.
#         **kwargs : dict, optional
#             Additional keyword arguments to be passed to the `xarray.Dataset.to_zarr` method.
#             Refer to the `xarray.Dataset.to_zarr <https://docs.xarray.dev/en/latest/generated/xarray.Dataset.to_zarr.html>`__
#             for available options.

#         Returns
#         -------
#         None

#         Raises
#         ------
#         OSError
#             If the specified store path is invalid or not writable.

#         Examples
#         --------
#         >>> # Save the Processing Set to a local Zarr store
#         >>> ps.to_store('/path/to/zarr_store')
#         """
#         import os

#         for key, value in self.items():
#             value.to_store(os.path.join(store, key), **kwargs)

#     def get_combined_field_and_source_xds(self, data_group="base"):
#         """
#         Combine all non-ephemeris `field_and_source_xds` datasets from a Processing Set for a datagroup into a single dataset.

#         Parameters
#         ----------
#         data_group : str, optional
#             The data group to process. Default is "base".

#         Returns
#         -------
#         xarray.Dataset
#             combined_field_and_source_xds: Combined dataset for standard fields.

#         Raises
#         ------
#         ValueError
#             If the `field_and_source_xds` attribute is missing or improperly formatted in any Measurement Set.
#         """

#         combined_field_and_source_xds = xr.Dataset()
#         for ms_name, ms_xds in self.items():
#             correlated_data_name = ms_xds.attrs["data_groups"][data_group][
#                 "correlated_data"
#             ]

#             field_and_source_xds = (
#                 ms_xds[correlated_data_name]
#                 .attrs["field_and_source_xds"]
#                 .copy(deep=True)
#             )

#             if not field_and_source_xds.attrs["type"] == "field_and_source_ephemeris":

#                 if (
#                     "line_name" in field_and_source_xds.coords
#                 ):  # Not including line info since it is a function of spw.
#                     field_and_source_xds = field_and_source_xds.drop_vars(
#                         ["LINE_REST_FREQUENCY", "LINE_SYSTEMIC_VELOCITY"],
#                         errors="ignore",
#                     )
#                     del field_and_source_xds["line_name"]
#                     del field_and_source_xds["line_label"]

#                 if len(combined_field_and_source_xds.data_vars) == 0:
#                     combined_field_and_source_xds = field_and_source_xds
#                 else:
#                     combined_field_and_source_xds = xr.concat(
#                         [combined_field_and_source_xds, field_and_source_xds],
#                         dim="field_name",
#                     )

#         if (len(combined_field_and_source_xds.data_vars) > 0) and (
#             "FIELD_PHASE_CENTER" in combined_field_and_source_xds
#         ):
#             combined_field_and_source_xds = (
#                 combined_field_and_source_xds.drop_duplicates("field_name")
#             )

#             combined_field_and_source_xds["MEAN_PHASE_CENTER"] = (
#                 combined_field_and_source_xds["FIELD_PHASE_CENTER"].mean(
#                     dim=["field_name"]
#                 )
#             )

#             ra1 = (
#                 combined_field_and_source_xds["FIELD_PHASE_CENTER"]
#                 .sel(sky_dir_label="ra")
#                 .values
#             )
#             dec1 = (
#                 combined_field_and_source_xds["FIELD_PHASE_CENTER"]
#                 .sel(sky_dir_label="dec")
#                 .values
#             )
#             ra2 = (
#                 combined_field_and_source_xds["MEAN_PHASE_CENTER"]
#                 .sel(sky_dir_label="ra")
#                 .values
#             )
#             dec2 = (
#                 combined_field_and_source_xds["MEAN_PHASE_CENTER"]
#                 .sel(sky_dir_label="dec")
#                 .values
#             )

#             from xradio._utils.coord_math import haversine

#             distance = haversine(ra1, dec1, ra2, dec2)
#             min_index = distance.argmin()

#             combined_field_and_source_xds.attrs["center_field_name"] = (
#                 combined_field_and_source_xds.field_name[min_index].values
#             )

#         return combined_field_and_source_xds


#     def get_combined_field_and_source_xds_ephemeris(self, data_group="base"):
#         """
#         Combine all ephemeris `field_and_source_xds` datasets from a Processing Set for a datagroup into a single dataset.

#         Parameters
#         ----------
#         data_group : str, optional
#             The data group to process. Default is "base".

#         Returns
#         -------
#         xarray.Dataset
#             - combined_ephemeris_field_and_source_xds: Combined dataset for ephemeris fields.

#         Raises
#         ------
#         ValueError
#             If the `field_and_source_xds` attribute is missing or improperly formatted in any Measurement Set.
#         """

#         combined_ephemeris_field_and_source_xds = xr.Dataset()
#         for ms_name, ms_xds in self.items():

#             correlated_data_name = ms_xds.attrs["data_groups"][data_group][
#                 "correlated_data"
#             ]

#             field_and_source_xds = (
#                 ms_xds[correlated_data_name]
#                 .attrs["field_and_source_xds"]
#                 .copy(deep=True)
#             )

#             if field_and_source_xds.attrs["type"] == "field_and_source_ephemeris":

#                 if (
#                     "line_name" in field_and_source_xds.coords
#                 ):  # Not including line info since it is a function of spw.
#                     field_and_source_xds = field_and_source_xds.drop_vars(
#                         ["LINE_REST_FREQUENCY", "LINE_SYSTEMIC_VELOCITY"],
#                         errors="ignore",
#                     )
#                     del field_and_source_xds["line_name"]
#                     del field_and_source_xds["line_label"]

#                 from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
#                     interpolate_to_time,
#                 )

#                 if "time_ephemeris" in field_and_source_xds:
#                     field_and_source_xds = interpolate_to_time(
#                         field_and_source_xds,
#                         field_and_source_xds.time,
#                         "field_and_source_xds",
#                         "time_ephemeris",
#                     )
#                     del field_and_source_xds["time_ephemeris"]
#                     field_and_source_xds = field_and_source_xds.rename(
#                         {"time_ephemeris": "time"}
#                     )

#                 if "OBSERVER_POSITION" in field_and_source_xds:
#                     field_and_source_xds = field_and_source_xds.drop_vars(
#                         ["OBSERVER_POSITION"], errors="ignore"
#                     )

#                 if len(combined_ephemeris_field_and_source_xds.data_vars) == 0:
#                     combined_ephemeris_field_and_source_xds = field_and_source_xds
#                 else:

#                     combined_ephemeris_field_and_source_xds = xr.concat(
#                         [combined_ephemeris_field_and_source_xds, field_and_source_xds],
#                         dim="time",
#                     )

#         if (len(combined_ephemeris_field_and_source_xds.data_vars) > 0) and (
#             "FIELD_PHASE_CENTER" in combined_ephemeris_field_and_source_xds
#         ):

#             from xradio._utils.coord_math import wrap_to_pi

#             offset = (
#                 combined_ephemeris_field_and_source_xds["FIELD_PHASE_CENTER"]
#                 - combined_ephemeris_field_and_source_xds["SOURCE_LOCATION"]
#             )
#             combined_ephemeris_field_and_source_xds["FIELD_OFFSET"] = xr.DataArray(
#                 wrap_to_pi(offset.sel(sky_pos_label=["ra", "dec"])).values,
#                 dims=["time", "sky_dir_label"],
#             )
#             combined_ephemeris_field_and_source_xds["FIELD_OFFSET"].attrs = (
#                 combined_ephemeris_field_and_source_xds["FIELD_PHASE_CENTER"].attrs
#             )
#             combined_ephemeris_field_and_source_xds["FIELD_OFFSET"].attrs["units"] = (
#                 combined_ephemeris_field_and_source_xds["FIELD_OFFSET"].attrs["units"][
#                     :2
#                 ]
#             )

#             ra1 = (
#                 combined_ephemeris_field_and_source_xds["FIELD_OFFSET"]
#                 .sel(sky_dir_label="ra")
#                 .values
#             )
#             dec1 = (
#                 combined_ephemeris_field_and_source_xds["FIELD_OFFSET"]
#                 .sel(sky_dir_label="dec")
#                 .values
#             )
#             ra2 = 0.0
#             dec2 = 0.0

#             from xradio._utils.coord_math import haversine

#             distance = haversine(ra1, dec1, ra2, dec2)
#             min_index = distance.argmin()

#             combined_ephemeris_field_and_source_xds.attrs["center_field_name"] = (
#                 combined_ephemeris_field_and_source_xds.field_name[min_index].values
#             )

#         return combined_ephemeris_field_and_source_xds

#     def plot_phase_centers(self, label_all_fields=False, data_group="base"):
#         """
#         Plot the phase center locations of all fields in the Processing Set.

#         This method is primarily used for visualizing mosaics. It generates scatter plots of
#         the phase center coordinates for both standard and ephemeris fields. The central field
#         is highlighted in red based on the closest phase center calculation.

#         Parameters
#         ----------
#         label_all_fields : bool, optional
#             If `True`, all fields will be labeled on the plot. Default is `False`.
#         data_group : str, optional
#             The data group to use for processing. Default is "base".

#         Returns
#         -------
#         None

#         Raises
#         ------
#         ValueError
#             If the combined datasets are empty or improperly formatted.
#         """
#         combined_field_and_source_xds = self.get_combined_field_and_source_xds(
#             data_group
#         )
#         combined_ephemeris_field_and_source_xds = (
#             self.get_combined_field_and_source_xds_ephemeris(data_group)
#         )
#         from matplotlib import pyplot as plt

#         if (len(combined_field_and_source_xds.data_vars) > 0) and (
#             "FIELD_PHASE_CENTER" in combined_field_and_source_xds
#         ):
#             plt.figure()
#             plt.title("Field Phase Center Locations")
#             plt.scatter(
#                 combined_field_and_source_xds["FIELD_PHASE_CENTER"].sel(
#                     sky_dir_label="ra"
#                 ),
#                 combined_field_and_source_xds["FIELD_PHASE_CENTER"].sel(
#                     sky_dir_label="dec"
#                 ),
#             )

#             center_field_name = combined_field_and_source_xds.attrs["center_field_name"]
#             center_field = combined_field_and_source_xds.sel(
#                 field_name=center_field_name
#             )
#             plt.scatter(
#                 center_field["FIELD_PHASE_CENTER"].sel(sky_dir_label="ra"),
#                 center_field["FIELD_PHASE_CENTER"].sel(sky_dir_label="dec"),
#                 color="red",
#                 label=center_field_name,
#             )
#             plt.xlabel("RA (rad)")
#             plt.ylabel("DEC (rad)")
#             plt.legend()
#             plt.show()

#         if (len(combined_ephemeris_field_and_source_xds.data_vars) > 0) and (
#             "FIELD_PHASE_CENTER" in combined_ephemeris_field_and_source_xds
#         ):

#             plt.figure()
#             plt.title(
#                 "Offset of Field Phase Center from Source Location (Ephemeris Data)"
#             )
#             plt.scatter(
#                 combined_ephemeris_field_and_source_xds["FIELD_OFFSET"].sel(
#                     sky_dir_label="ra"
#                 ),
#                 combined_ephemeris_field_and_source_xds["FIELD_OFFSET"].sel(
#                     sky_dir_label="dec"
#                 ),
#             )

#             center_field_name = combined_ephemeris_field_and_source_xds.attrs[
#                 "center_field_name"
#             ]

#             combined_ephemeris_field_and_source_xds = (
#                 combined_ephemeris_field_and_source_xds.set_xindex("field_name")
#             )

#             center_field = combined_ephemeris_field_and_source_xds.sel(
#                 field_name=center_field_name
#             )
#             plt.scatter(
#                 center_field["FIELD_OFFSET"].sel(sky_dir_label="ra"),
#                 center_field["FIELD_OFFSET"].sel(sky_dir_label="dec"),
#                 color="red",
#                 label=center_field_name,
#             )
#             plt.xlabel("RA Offset (rad)")
#             plt.ylabel("DEC Offset (rad)")
#             plt.legend()
#             plt.show()

#     def get_combined_antenna_xds(self):
#         """
#         Combine the `antenna_xds` datasets from all Measurement Sets into a single dataset.

#         This method concatenates the antenna datasets from each Measurement Set along the 'antenna_name' dimension.

#         Returns
#         -------
#         xarray.Dataset
#             A combined `xarray.Dataset` containing antenna information from all Measurement Sets.

#         Raises
#         ------
#         ValueError
#             If antenna datasets are missing required variables or improperly formatted.
#         """
#         combined_antenna_xds = xr.Dataset()
#         for cor_name, ms_xds in self.items():
#             antenna_xds = ms_xds.antenna_xds.copy(deep=True)

#             if len(combined_antenna_xds.data_vars) == 0:
#                 combined_antenna_xds = antenna_xds
#             else:
#                 combined_antenna_xds = xr.concat(
#                     [combined_antenna_xds, antenna_xds],
#                     dim="antenna_name",
#                     data_vars="minimal",
#                     coords="minimal",
#                 )

#         # ALMA WVR antenna_xds data has a NaN value for the antenna receptor angle.
#         if "ANTENNA_RECEPTOR_ANGLE" in combined_antenna_xds.data_vars:
#             combined_antenna_xds = combined_antenna_xds.dropna("antenna_name")

#         combined_antenna_xds = combined_antenna_xds.drop_duplicates("antenna_name")

#         return combined_antenna_xds

#     def plot_antenna_positions(self):
#         """
#         Plot the antenna positions of all antennas in the Processing Set.

#         This method generates three scatter plots displaying the antenna positions in different planes:
#         - X vs Y
#         - X vs Z
#         - Y vs Z

#         Parameters
#         ----------
#         None

#         Returns
#         -------
#         None

#         Raises
#         ------
#         ValueError
#             If the combined antenna dataset is empty or missing required coordinates.
#         """
#         combined_antenna_xds = self.get_combined_antenna_xds()
#         from matplotlib import pyplot as plt

#         plt.figure()
#         plt.title("Antenna Positions")
#         plt.scatter(
#             combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="x"),
#             combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="y"),
#         )
#         plt.xlabel("x (m)")
#         plt.ylabel("y (m)")
#         plt.show()

#         plt.figure()
#         plt.title("Antenna Positions")
#         plt.scatter(
#             combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="x"),
#             combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="z"),
#         )
#         plt.xlabel("x (m)")
#         plt.ylabel("z (m)")
#         plt.show()

#         plt.figure()
#         plt.title("Antenna Positions")
#         plt.scatter(
#             combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="y"),
#             combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="z"),
#         )
#         plt.xlabel("y (m)")
#         plt.ylabel("z (m)")
#         plt.show()
