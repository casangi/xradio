import pandas as pd
from xradio._utils.list_and_array import to_list
import numpy as np
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

    def summary(self, data_group: str = None) -> pd.DataFrame:
        """
        Generate and retrieve a summary of the Processing Set.

        The summary includes information such as the names of the Measurement Sets,
        their intents, polarizations, spectral window names, field names, source names,
        field coordinates, start frequencies, and end frequencies.

        Parameters
        ----------
        data_group : str, optional
            The data group to summarize. By default the "base" group
            is used (if found), or otherwise the first group found.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the summary information of the specified data group.
        """

        def find_data_group_base_or_first(data_group: str, xdt: xr.DataTree) -> str:
            first_msv4 = next(iter(xdt.values()))
            first_data_groups = first_msv4.attrs["data_groups"]
            if data_group is None:
                data_group = (
                    "base"
                    if "base" in first_data_groups
                    else next(iter(first_data_groups))
                )
            return data_group

        if self._xdt.attrs.get("type") not in PS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a processing set node."
            )

        data_group = find_data_group_base_or_first(data_group, self._xdt)

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
            spw_names = []
            freq_axis_list = []
            frame = self._xdt[next(iter(self._xdt.children))].frequency.attrs[
                "observer"
            ]
            for ms_xdt in self._xdt.values():
                assert (
                    frame == ms_xdt.frequency.attrs["observer"]
                ), "Frequency reference frame not consistent in Processing Set."
                if ms_xdt.frequency.attrs["spectral_window_name"] not in spw_names:
                    spw_names.append(ms_xdt.frequency.attrs["spectral_window_name"])
                    freq_axis_list.append(ms_xdt.frequency)

            freq_axis = xr.concat(freq_axis_list, dim="frequency", join="outer").sortby(
                "frequency"
            )
            self.meta["freq_axis"] = freq_axis
            return self.meta["freq_axis"]

    def _summary(self, data_group: str = None):
        summary_data = {
            "name": [],
            "intents": [],
            "shape": [],
            "polarization": [],
            "scan_name": [],
            "spw_name": [],
            "spw_intent": [],
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
            summary_data["spw_intent"].append(partition_info["spectral_window_intent"])
            summary_data["polarization"].append(value.polarization.values)
            summary_data["scan_name"].append(partition_info["scan_name"])
            data_name = value.attrs["data_groups"][data_group]["correlated_data"]

            if "VISIBILITY" in data_name:
                center_name = "FIELD_PHASE_CENTER_DIRECTION"

            if "SPECTRUM" in data_name:
                center_name = "FIELD_REFERENCE_CENTER_DIRECTION"

            summary_data["shape"].append(value[data_name].shape)

            summary_data["field_name"].append(partition_info["field_name"])
            summary_data["source_name"].append(partition_info["source_name"])

            summary_data["line_name"].append(partition_info["line_name"])

            summary_data["start_frequency"].append(
                to_list(value["frequency"].values)[0]
            )
            summary_data["end_frequency"].append(to_list(value["frequency"].values)[-1])

            field_and_source_xds = value.xr_ms.get_field_and_source_xds(data_group)

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
        >>> selected_ps_xdt = ps_xdt.xr_ps.query(intents='OBSERVE_TARGET#ON_SOURCE', polarization=['RR', 'LL'])

        >>> # Select all MSs with start_frequency greater than 100 GHz and less than 200 GHz
        >>> selected_ps_xdt = ps_xdt.xr_ps.query(query='start_frequency > 100e9 AND end_frequency < 200e9')
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
            field_and_source_xds = ms_xdt.xr_ms.get_field_and_source_xds(data_group)

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
                        join="outer",
                    )

        if (len(combined_field_and_source_xds.data_vars) > 0) and (
            "FIELD_PHASE_CENTER_DIRECTION" in combined_field_and_source_xds
        ):
            combined_field_and_source_xds = (
                combined_field_and_source_xds.drop_duplicates("field_name")
            )

            combined_field_and_source_xds["MEAN_PHASE_CENTER_DIRECTION"] = (
                combined_field_and_source_xds["FIELD_PHASE_CENTER_DIRECTION"].mean(
                    dim=["field_name"]
                )
            )

            ra1 = (
                combined_field_and_source_xds["FIELD_PHASE_CENTER_DIRECTION"]
                .sel(sky_dir_label="ra")
                .values
            )
            dec1 = (
                combined_field_and_source_xds["FIELD_PHASE_CENTER_DIRECTION"]
                .sel(sky_dir_label="dec")
                .values
            )
            ra2 = (
                combined_field_and_source_xds["MEAN_PHASE_CENTER_DIRECTION"]
                .sel(sky_dir_label="ra")
                .values
            )
            dec2 = (
                combined_field_and_source_xds["MEAN_PHASE_CENTER_DIRECTION"]
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
            field_and_source_xds = field_and_source_xds = (
                ms_xdt.xr_ms.get_field_and_source_xds(data_group)
            )

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

                from xradio.measurement_set._utils._utils.interpolate import (
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
                        join="outer",
                    )

        if (len(combined_ephemeris_field_and_source_xds.data_vars) > 0) and (
            "FIELD_PHASE_CENTER_DIRECTION" in combined_ephemeris_field_and_source_xds
        ):

            from xradio._utils.coord_math import wrap_to_pi

            offset = (
                combined_ephemeris_field_and_source_xds["FIELD_PHASE_CENTER_DIRECTION"]
                - combined_ephemeris_field_and_source_xds["SOURCE_DIRECTION"]
            )
            combined_ephemeris_field_and_source_xds["FIELD_OFFSET"] = xr.DataArray(
                wrap_to_pi(offset.sel(sky_dir_label=["ra", "dec"])).values,
                dims=["time", "sky_dir_label"],
            )
            combined_ephemeris_field_and_source_xds["FIELD_OFFSET"].attrs = (
                combined_ephemeris_field_and_source_xds[
                    "FIELD_PHASE_CENTER_DIRECTION"
                ].attrs
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

        def setup_annotations_all(axis, scatter, field_names):
            """
            Creates annotations for when label_all_fields=True
            """
            coord_x, coord_y = np.array(scatter.get_offsets()).transpose()
            offset_x = np.abs(np.max(coord_x) - np.min(coord_x)) * 0.01
            offset_y = np.abs(np.max(coord_y) - np.min(coord_y)) * 0.01
            for idx, (x, y) in enumerate(zip(coord_x + offset_x, coord_y + offset_y)):
                axis.annotate(field_names[idx], (x, y), alpha=1)

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
            "FIELD_PHASE_CENTER_DIRECTION" in combined_field_and_source_xds
        ):
            fig = plt.figure()
            plt.title("Field Phase Center Locations")
            scatter = plt.scatter(
                combined_field_and_source_xds["FIELD_PHASE_CENTER_DIRECTION"].sel(
                    sky_dir_label="ra"
                ),
                combined_field_and_source_xds["FIELD_PHASE_CENTER_DIRECTION"].sel(
                    sky_dir_label="dec"
                ),
            )
            center_field_name = combined_field_and_source_xds.attrs["center_field_name"]
            center_field = combined_field_and_source_xds.sel(
                field_name=center_field_name
            )

            if label_all_fields:
                field_name = combined_field_and_source_xds.field_name.values
                setup_annotations_all(fig.axes[0], scatter, field_name)
                fig.axes[0].margins(0.2, 0.2)
                center_label = None
            else:
                center_label = center_field_name

            plt.scatter(
                center_field["FIELD_PHASE_CENTER_DIRECTION"].sel(sky_dir_label="ra"),
                center_field["FIELD_PHASE_CENTER_DIRECTION"].sel(sky_dir_label="dec"),
                color="red",
                label=center_label,
            )
            plt.xlabel("RA (rad)")
            plt.ylabel("DEC (rad)")
            if not label_all_fields:
                plt.legend()
            plt.show()

        if (len(combined_ephemeris_field_and_source_xds.data_vars) > 0) and (
            "FIELD_PHASE_CENTER_DIRECTION" in combined_ephemeris_field_and_source_xds
        ):

            fig = plt.figure()
            plt.title(
                "Offset of Field Phase Center from Source Location (Ephemeris Data)"
            )
            scatter = plt.scatter(
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

            if label_all_fields:
                field_name = combined_ephemeris_field_and_source_xds.field_name.values
                setup_annotations_all(fig.axes[0], scatter, field_name)
                fig.axes[0].margins(0.2, 0.2)
                center_label = None
            else:
                center_label = center_field_name

            plt.scatter(
                center_field["FIELD_OFFSET"].sel(sky_dir_label="ra"),
                center_field["FIELD_OFFSET"].sel(sky_dir_label="dec"),
                color="red",
                label=center_label,
            )

            plt.xlabel("RA Offset (rad)")
            plt.ylabel("DEC Offset (rad)")
            if not label_all_fields:
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
                    join="outer",
                )

        # ALMA WVR antenna_xds data has a NaN value for the antenna receptor angle.
        if "ANTENNA_RECEPTOR_ANGLE" in combined_antenna_xds.data_vars:
            combined_antenna_xds = combined_antenna_xds.dropna("antenna_name")

        combined_antenna_xds = combined_antenna_xds.drop_duplicates("antenna_name")

        return combined_antenna_xds

    def plot_antenna_positions(self, label_all_antennas: bool = False):
        """
        Plot the antenna positions of all antennas in the Processing Set.

        This method generates and displays a figure with three scatter plots, displaying the antenna
        positions in different planes:

        - X vs Y
        - X vs Z
        - Y vs Z

        The antenna names are shown on hovering their positions, unless label_all_antennas is enabled.

        Parameters
        ----------
        label_all_antennas : bool, optional
            If 'True', annotations are shown with the names of every antenna next to their positions.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the combined antenna dataset is empty or missing required coordinates.
        """

        def antenna_hover(event):
            if event.inaxes in antenna_axes:
                for axis in antenna_axes:
                    contained, indices = scatter_map[axis].contains(event)
                    annotation = annotations_map[axis]
                    if contained:
                        scatter = scatter_map[axis]
                        update_antenna_annotation(indices, scatter, annotation)
                        annotation.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        visible = annotation.get_visible()
                        if visible:
                            annotation.set_visible(False)
                            fig.canvas.draw_idle()

        def update_antenna_annotation(indices, scatter, annotation):
            position = scatter.get_offsets()[indices["ind"][0]]
            annotation.xy = position
            text = "{}".format(" ".join([antenna_names[num] for num in indices["ind"]]))
            annotation.set_text(text)
            annotation.get_bbox_patch().set_facecolor("#e8d192")
            annotation.get_bbox_patch().set_alpha(1)

        def setup_annotations_for_hover(antenna_axes, scatter_plots):
            """
            Creates annotations on all the axes requested.

            Returns
            -------
            dict
                dict from antenna axes -> annotation objects
            """
            antenna_annotations = []
            for axis in antenna_axes:
                annotation = axis.annotate(
                    "",
                    xy=(0, 0),
                    xytext=(10, 15),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-|>"),
                    bbox=dict(boxstyle="round", fc="w"),
                )
                antenna_annotations.append(annotation)
                annotation.set_visible(False)
            annotations_map = dict(zip(antenna_axes, antenna_annotations))

            return annotations_map

        def setup_annotations_for_all(antenna_axes, scatter_map):
            """
            Creates annotations for when label_all_antennas=True
            """
            for axis in antenna_axes:
                scatter = scatter_map[axis]
                coord_x, coord_y = np.array(scatter.get_offsets()).transpose()
                offset_x = np.abs(np.max(coord_x) - np.min(coord_x)) * 0.01
                offset_y = np.abs(np.max(coord_y) - np.min(coord_y)) * 0.01
                for idx, (x, y) in enumerate(
                    zip(coord_x + offset_x, coord_y + offset_y)
                ):
                    axis.annotate(
                        antenna_names[idx],
                        (x, y),
                        alpha=1,
                    )

        if self._xdt.attrs.get("type") not in PS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a processing set node."
            )

        combined_antenna_xds = self.get_combined_antenna_xds()
        from matplotlib import pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Antenna Positions")
        fig.subplots_adjust(
            wspace=0.25, hspace=0.25, left=0.1, right=0.95, top=0.9, bottom=0.1
        )

        scatter1 = ax1.scatter(
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="x"),
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="y"),
        )
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        antenna_names = combined_antenna_xds.antenna_name.values

        scatter2 = ax2.scatter(
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="y"),
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="z"),
        )
        ax2.set_xlabel("y (m)")
        ax2.set_ylabel("z (m)")

        scatter3 = ax3.scatter(
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="x"),
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="z"),
        )
        ax3.set_xlabel("x (m)")
        ax3.set_ylabel("z (m)")

        ax4.axis("off")

        antenna_axes = [ax1, ax2, ax3]
        scatter_map = dict(zip(antenna_axes, [scatter1, scatter2, scatter3]))
        if label_all_antennas:
            annotations_map = setup_annotations_for_all(antenna_axes, scatter_map)
        else:
            annotations_map = setup_annotations_for_hover(antenna_axes, scatter_map)
            fig.canvas.mpl_connect("motion_notify_event", antenna_hover)

        plt.show()
