import pandas as pd
from xradio._utils.list_and_array import to_list
import numbers
import numpy as np
import toolviper.utils.logger as logger
import xarray as xr


class ProcessingSet(dict):
    """
    A dictionary subclass representing a Processing Set (PS) containing Measurement Sets v4 (MS).

    This class extends the built-in `dict` class to provide additional methods for
    manipulating and selecting subsets of the Processing Set. It includes functionality
    for summarizing metadata, selecting subsets based on various criteria, and
    exporting the data to storage formats.

    Parameters
    ----------
    *args : dict, optional
        Variable length argument list passed to the base `dict` class.
    **kwargs : dict, optional
        Arbitrary keyword arguments passed to the base `dict` class.

    Attributes
    ----------
    _meta : dict
        A dictionary for storing metadata related to the Processing Set. It includes keys such as
        "summary", "max_dims", and "freq_axis" to cache summary tables, maximum dimensions, and
        frequency axes respectively.

    Methods
    -------
    summary(data_group="base")
        Returns a summary of the Processing Set as a Pandas DataFrame.
    get_ps_max_dims()
        Retrieves the maximum dimensions across all Measurement Sets in the Processing Set.
    get_ps_freq_axis()
        Combines the frequency axes of all Measurement Sets in the Processing Set.
    sel(string_exact_match=True, query=None, **kwargs)
        Selects a subset of the Processing Set based on column names and values or a Pandas query.
    ms_sel(**kwargs)
        Applies the `xarray.Dataset.sel` method to each Measurement Set in the Processing Set to select a subset.
    ms_isel(**kwargs)
        Applies the `isel` method to each Measurement Set in the Processing Set to select by index.
    to_store(store, **kwargs)
        Writes the Processing Set to a Zarr store.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the ProcessingSet instance.

        Parameters
        ----------
        *args : dict, optional
            Variable length argument list passed to the base `dict` class.
        **kwargs : dict, optional
            Arbitrary keyword arguments passed to the base `dict` class.
        """
        super().__init__(*args, **kwargs)
        self._meta = {"summary": {}}

    def summary(self, data_group="base"):
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

        Raises
        ------
        KeyError
            If the specified `data_group` does not exist in the metadata and the summary cannot be generated.
        """
        if data_group in self._meta["summary"]:
            return self._meta["summary"][data_group]
        else:
            self._meta["summary"][data_group] = self._summary(data_group).sort_values(
                by=["name"], ascending=True
            )
            return self._meta["summary"][data_group]

    def get_ps_max_dims(self):
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
        if "max_dims" in self._meta:
            return self._meta["max_dims"]
        else:
            self._meta["max_dims"] = self._get_ps_max_dims()
            return self._meta["max_dims"]

    def get_ps_freq_axis(self):
        """
        Combine the frequency axes of all Measurement Sets in the Processing Set.

        This method aggregates the frequency information from each Measurement Set to create
        a unified frequency axis for the entire Processing Set.

        Returns
        -------
        xarray.DataArray
            The combined frequency axis of the Processing Set.
        """
        if "freq_axis" in self._meta:
            return self._meta["freq_axis"]
        else:
            self._meta["freq_axis"] = self._get_ps_freq_axis()
            return self._meta["freq_axis"]

    def sel(self, string_exact_match: bool = True, query: str = None, **kwargs):
        """
        Select a subset of the Processing Set based on specified criteria.

        This method allows filtering the Processing Set by matching column names and values
        or by applying a Pandas query string. The selection criteria can target various
        attributes of the Measurement Sets such as intents, polarization, spectral window names, etc.

        Note
        ----
        This selection does not modify the actual data within the Measurement Sets. For example, if
        a Measurement Set has `field_name=['field_0','field_10','field_08']` and `ps.sel(field_name='field_0')`
        is invoked, the resulting subset will still contain the original list `['field_0','field_10','field_08']`.

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
        ProcessingSet
            A new `ProcessingSet` instance containing only the Measurement Sets that match the selection criteria.

        Examples
        --------
        >>> # Select all MSs with intents 'OBSERVE_TARGET#ON_SOURCE' and polarization 'RR' or 'LL'
        >>> selected_ps = ps.sel(intents='OBSERVE_TARGET#ON_SOURCE', polarization=['RR', 'LL'])

        >>> # Select all MSs with start_frequency greater than 100 GHz and less than 200 GHz
        >>> selected_ps = ps.sel(query='start_frequency > 100e9 AND end_frequency < 200e9')
        """
        import numpy as np

        def select_rows(df, col, sel_vals, string_exact_match):
            def check_selection(row_val):
                row_val = to_list(row_val)  # Ensure it's a list for iteration

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
            value = to_list(value)  # Ensure value is a list

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
        Select a subset of the Processing Set by applying the `xarray.Dataset.sel` method to each Measurement Set.

        This method allows for selection based on label-based indexing for each dimension of the datasets.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments representing dimension names and the labels to select along those dimensions.
            These are passed directly to the `xarray.Dataset.sel` method.

        Returns
        -------
        ProcessingSet
            A new `ProcessingSet` instance containing the selected subsets of each Measurement Set.

        Examples
        --------
        >>> # Select data where time is '2024-01-01'
        >>> selected_ps = ps.ms_sel(time='2024-01-01')

        >>> # Select specific latitude and longitude
        >>> selected_ps = ps.ms_sel(latitude=45.0, longitude=-120.0)
        """
        sub_ps = ProcessingSet()
        for key, val in self.items():
            sub_ps[key] = val.sel(**kwargs)
        return sub_ps

    def ms_isel(self, **kwargs):
        """
        Select a subset of the Processing Set by applying the `isel` method to each Measurement Set.

        This method allows for selection based on integer-based indexing for each dimension of the datasets.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments representing dimension names and the integer indices to select along those dimensions.
            These are passed directly to the `isel` method.

        Returns
        -------
        ProcessingSet
            A new `ProcessingSet` instance containing the selected subsets of each Measurement Set.

        Examples
        --------
        >>> # Select the first time index
        >>> selected_ps = ps.ms_isel(time=0)

        >>> # Select multiple indices for latitude and longitude
        >>> selected_ps = ps.ms_isel(latitude=[0, 2, 4], longitude=1)
        """
        sub_ps = ProcessingSet()
        for key, val in self.items():
            sub_ps[key] = val.isel(**kwargs)
        return sub_ps

    def to_store(self, store, **kwargs):
        """
        Write the Processing Set to a Zarr store.

        This method serializes each Measurement Set within the Processing Set to a separate Zarr group
        within the specified store directory. Note that writing to cloud storage is not supported yet.

        Parameters
        ----------
        store : str
            The filesystem path to the Zarr store directory where the data will be saved.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the `xarray.Dataset.to_zarr` method.
            Refer to the [xarray documentation](https://docs.xarray.dev/en/latest/generated/xarray.Dataset.to_zarr.html)
            for available options.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If the specified store path is invalid or not writable.

        Examples
        --------
        >>> # Save the Processing Set to a local Zarr store
        >>> ps.to_store('/path/to/zarr_store')

        """
        import os

        for key, value in self.items():
            value.to_zarr(os.path.join(store, key), **kwargs)

        import pandas as pd


from xradio._utils.list_and_array import to_list
import numbers
import numpy as np
import toolviper.utils.logger as logger
import xarray as xr
from astropy.coordinates import SkyCoord
import astropy.units as u


class ProcessingSet(dict):
    """
    A dictionary subclass representing a Processing Set (PS) containing Measurement Sets v4 (MS).

    This class extends the built-in `dict` class to provide additional methods for
    manipulating and selecting subsets of the Processing Set. It includes functionality
    for summarizing metadata, selecting subsets based on various criteria, and
    exporting the data to storage formats.

    Parameters
    ----------
    *args : dict, optional
        Variable length argument list passed to the base `dict` class.
    **kwargs : dict, optional
        Arbitrary keyword arguments passed to the base `dict` class.

    Methods
    -------
    summary(data_group="base")
        Returns a summary of the Processing Set as a Pandas DataFrame.
    get_ps_max_dims()
        Retrieves the maximum dimensions across all Measurement Sets in the Processing Set.
    get_ps_freq_axis()
        Combines the frequency axes of all Measurement Sets in the Processing Set.
    sel(string_exact_match=True, query=None, **kwargs)
        Selects a subset of the Processing Set based on column names and values or a Pandas query.
    ms_sel(**kwargs)
        Applies the `xarray.Dataset.sel` method to each Measurement Set in the Processing Set to select a subset.
    ms_isel(**kwargs)
        Applies the `isel` method to each Measurement Set in the Processing Set to select by index.
    to_store(store, **kwargs)
        Writes the Processing Set to a Zarr store.
    get_combined_field_and_source_xds(data_group="base")
        Combines field and source datasets from all Measurement Sets into a single dataset.
    plot_phase_centers(label_all_fields=False, data_group="base")
        Plots the phase center locations of all fields in the Processing Set.
    get_combined_antenna_xds()
        Combines antenna datasets from all Measurement Sets into a single dataset.
    plot_antenna_positions()
        Plots the antenna positions of all antennas in the Processing Set.
    _get(id)
        Retrieves a Measurement Set by its index.
    _summary(data_group="base")
        Generates a summary DataFrame for all Measurement Sets in the Processing Set.
    _get_ps_freq_axis()
        Combines the frequency axes from all spectral windows into a single frequency axis.
    _get_ps_max_dims()
        Determines the maximum dimension sizes across all Measurement Sets.

    Examples
    --------
    >>> ps = ProcessingSet()
    >>> ps['ms1'] = xr.Dataset(...)
    >>> ps['ms2'] = xr.Dataset(...)
    >>> summary_df = ps.summary()
    >>> max_dims = ps.get_ps_max_dims()
    >>> selected_ps = ps.sel(intents='OBSERVE_TARGET#ON_SOURCE', polarization=['RR', 'LL'])
    >>> ps.to_store('/path/to/store')
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the ProcessingSet instance.

        Parameters
        ----------
        *args : dict, optional
            Variable length argument list passed to the base `dict` class.
        **kwargs : dict, optional
            Arbitrary keyword arguments passed to the base `dict` class.
        """
        super().__init__(*args, **kwargs)
        self._meta = {"summary": {}}

    def summary(self, data_group="base"):
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

        Raises
        ------
        KeyError
            If the specified `data_group` does not exist in the metadata and the summary cannot be generated.
        """
        if data_group in self._meta["summary"]:
            return self._meta["summary"][data_group]
        else:
            self._meta["summary"][data_group] = self._summary(data_group).sort_values(
                by=["name"], ascending=True
            )
            return self._meta["summary"][data_group]

    def get_ps_max_dims(self):
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
        if "max_dims" in self._meta:
            return self._meta["max_dims"]
        else:
            self._meta["max_dims"] = self._get_ps_max_dims()
            return self._meta["max_dims"]

    def get_ps_freq_axis(self):
        """
        Combine the frequency axes of all Measurement Sets in the Processing Set.

        This method aggregates the frequency information from each Measurement Set to create
        a unified frequency axis for the entire Processing Set.

        Returns
        -------
        xarray.DataArray
            The combined frequency axis of the Processing Set.
        """
        if "freq_axis" in self._meta:
            return self._meta["freq_axis"]
        else:
            self._meta["freq_axis"] = self._get_ps_freq_axis()
            return self._meta["freq_axis"]

    def sel(self, string_exact_match: bool = True, query: str = None, **kwargs):
        """
        Select a subset of the Processing Set based on specified criteria.

        This method allows filtering the Processing Set by matching column names and values
        or by applying a Pandas query string. The selection criteria can target various
        attributes of the Measurement Sets such as intents, polarization, spectral window names, etc.

        Note
        ----
        This selection does not modify the actual data within the Measurement Sets. For example, if
        a Measurement Set has `field_name=['field_0','field_10','field_08']` and `ps.sel(field_name='field_0')`
        is invoked, the resulting subset will still contain the original list `['field_0','field_10','field_08']`.

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
        ProcessingSet
            A new `ProcessingSet` instance containing only the Measurement Sets that match the selection criteria.

        Examples
        --------
        >>> # Select all MSs with intents 'OBSERVE_TARGET#ON_SOURCE' and polarization 'RR' or 'LL'
        >>> selected_ps = ps.sel(intents='OBSERVE_TARGET#ON_SOURCE', polarization=['RR', 'LL'])

        >>> # Select all MSs with start_frequency greater than 100 GHz and less than 200 GHz
        >>> selected_ps = ps.sel(query='start_frequency > 100e9 AND end_frequency < 200e9')
        """
        import numpy as np

        def select_rows(df, col, sel_vals, string_exact_match):
            def check_selection(row_val):
                row_val = to_list(row_val)  # Ensure it's a list for iteration

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
            value = to_list(value)  # Ensure value is a list

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
        Select a subset of the Processing Set by applying the `xarray.Dataset.sel` method to each Measurement Set.

        This method allows for selection based on label-based indexing for each dimension of the datasets.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments representing dimension names and the labels to select along those dimensions.
            These are passed directly to the `xarray.Dataset.sel` method.

        Returns
        -------
        ProcessingSet
            A new `ProcessingSet` instance containing the selected subsets of each Measurement Set.

        Examples
        --------
        >>> # Select data where time is '2024-01-01'
        >>> selected_ps = ps.ms_sel(time='2024-01-01')

        >>> # Select specific latitude and longitude
        >>> selected_ps = ps.ms_sel(latitude=45.0, longitude=-120.0)
        """
        sub_ps = ProcessingSet()
        for key, val in self.items():
            sub_ps[key] = val.sel(**kwargs)
        return sub_ps

    def ms_isel(self, **kwargs):
        """
        Select a subset of the Processing Set by applying the `isel` method to each Measurement Set.

        This method allows for selection based on integer-based indexing for each dimension of the datasets.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments representing dimension names and the integer indices to select along those dimensions.
            These are passed directly to the `isel` method.

        Returns
        -------
        ProcessingSet
            A new `ProcessingSet` instance containing the selected subsets of each Measurement Set.

        Examples
        --------
        >>> # Select the first time index
        >>> selected_ps = ps.ms_isel(time=0)

        >>> # Select multiple indices for latitude and longitude
        >>> selected_ps = ps.ms_isel(latitude=[0, 2, 4], longitude=1)
        """
        sub_ps = ProcessingSet()
        for key, val in self.items():
            sub_ps[key] = val.isel(**kwargs)
        return sub_ps

    def to_store(self, store, **kwargs):
        """
        Write the Processing Set to a Zarr store.

        This method serializes each Measurement Set within the Processing Set to a separate Zarr group
        within the specified store directory. Note that writing to cloud storage is not supported yet.

        Parameters
        ----------
        store : str
            The filesystem path to the Zarr store directory where the data will be saved.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the `xarray.Dataset.to_zarr` method.
            Refer to the [xarray documentation](https://docs.xarray.dev/en/latest/generated/xarray.Dataset.to_zarr.html)
            for available options.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If the specified store path is invalid or not writable.

        Examples
        --------
        >>> # Save the Processing Set to a local Zarr store
        >>> ps.to_store('/path/to/zarr_store')

        >>> # Save with compression
        >>> ps.to_store('/path/to/zarr_store', encoding={'compression': 'gzip'})
        """
        import os

        for key, value in self.items():
            value.to_zarr(os.path.join(store, key), **kwargs)

    def get_combined_field_and_source_xds(self, data_group="base"):
        """
        Combine the `field_and_source_xds` datasets from all Measurement Sets into a single dataset.

        The combined `xarray.Dataset` will have a new dimension 'field_name', consolidating data from
        each Measurement Set. Ephemeris data is handled separately.

        Parameters
        ----------
        data_group : str, optional
            The data group to process. Default is "base".

        Returns
        -------
        tuple of xarray.Dataset
            A tuple containing two `xarray.Dataset` objects:
            - combined_field_and_source_xds: Combined dataset for standard fields.
            - combined_ephemeris_field_and_source_xds: Combined dataset for ephemeris fields.

        Raises
        ------
        ValueError
            If the `field_and_source_xds` attribute is missing or improperly formatted in any Measurement Set.

        Examples
        --------
        >>> combined_field_xds, combined_ephemeris_xds = ps.get_combined_field_and_source_xds()

        """
        df = self.summary(data_group)

        combined_field_and_source_xds = xr.Dataset()
        combined_ephemeris_field_and_source_xds = xr.Dataset()
        for ms_name, ms_xds in self.items():

            correlated_data_name = ms_xds.attrs["data_groups"][data_group][
                "correlated_data"
            ]

            field_and_source_xds = (
                ms_xds[correlated_data_name]
                .attrs["field_and_source_xds"]
                .copy(deep=True)
            )

            if "line_name" in field_and_source_xds.coords:
                # Not including line info since it is a function of spectral window (spw).
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
                        * len(field_and_source_xds.time_ephemeris.values)
                    )
                    source_names = np.array(
                        [field_and_source_xds.source_name.values.item()]
                        * len(field_and_source_xds.time_ephemeris.values)
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

        # Process standard fields
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

        # Process ephemeris fields
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

            distance = haversine(ra1, dec1, ra2, dec2)
            min_index = distance.argmin()

            combined_ephemeris_field_and_source_xds.attrs["center_field_name"] = (
                combined_ephemeris_field_and_source_xds.field_name[min_index].values
            )

        return combined_field_and_source_xds, combined_ephemeris_field_and_source_xds

    def plot_phase_centers(self, label_all_fields=False, data_group="base"):
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

        Examples
        --------
        >>> # Plot phase centers for the base data group
        >>> ps.plot_phase_centers()

        >>> # Plot phase centers and label all fields
        >>> ps.plot_phase_centers(label_all_fields=True)
        """
        combined_field_and_source_xds, combined_ephemeris_field_and_source_xds = (
            self.get_combined_field_and_source_xds(data_group)
        )
        from matplotlib import pyplot as plt

        # Plot standard field phase centers
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
                label="Fields",
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
                marker="x",
            )
            plt.xlabel("RA (rad)")
            plt.ylabel("DEC (rad)")
            plt.legend()
            plt.show()

        # Plot ephemeris field offsets
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
                label="Ephemeris Offsets",
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
                marker="x",
            )
            plt.xlabel("RA Offset (rad)")
            plt.ylabel("DEC Offset (rad)")
            plt.legend()
            plt.show()

    def get_combined_antenna_xds(self):
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

        Examples
        --------
        >>> combined_antenna_xds = ps.get_combined_antenna_xds()
        """
        combined_antenna_xds = xr.Dataset()
        for cor_name, ms_xds in self.items():
            antenna_xds = ms_xds.antenna_xds.copy(deep=True)

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

        Examples
        --------
        >>> ps.plot_antenna_positions()
        """
        combined_antenna_xds = self.get_combined_antenna_xds()
        from matplotlib import pyplot as plt

        # Plot X vs Y
        plt.figure()
        plt.title("Antenna Positions (X vs Y)")
        plt.scatter(
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="x"),
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="y"),
            label="Antenna Positions",
        )
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.legend()
        plt.show()

        # Plot X vs Z
        plt.figure()
        plt.title("Antenna Positions (X vs Z)")
        plt.scatter(
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="x"),
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="z"),
            label="Antenna Positions",
        )
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.legend()
        plt.show()

        # Plot Y vs Z
        plt.figure()
        plt.title("Antenna Positions (Y vs Z)")
        plt.scatter(
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="y"),
            combined_antenna_xds["ANTENNA_POSITION"].sel(cartesian_pos_label="z"),
            label="Antenna Positions",
        )
        plt.xlabel("y (m)")
        plt.ylabel("z (m)")
        plt.legend()
        plt.show()

    def _get(self, id):
        """
        Retrieve a Measurement Set by its index in the Processing Set.

        Parameters
        ----------
        id : int
            The index of the Measurement Set to retrieve.

        Returns
        -------
        xarray.Dataset
            The Measurement Set corresponding to the provided index.

        Raises
        ------
        IndexError
            If the provided index is out of range.
        """
        return self[list(self.keys())[id]]

    def _summary(self, data_group="base"):
        """
        Generate a summary DataFrame for all Measurement Sets in the Processing Set.

        This internal method collects metadata such as field names, polarization, frequency range,
        and phase center information for each Measurement Set and compiles them into a Pandas DataFrame.

        Parameters
        ----------
        data_group : str, optional
            The data group to summarize. Default is "base".

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing summary information for each Measurement Set.

        Raises
        ------
        KeyError
            If required attributes are missing in any Measurement Set.
        """
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
        """
        Combine the frequency axes from all spectral windows into a single frequency axis.

        This internal method concatenates the frequency data from each Measurement Set's spectral window
        and sorts them in ascending order. It ensures that all spectral windows share the same frequency
        reference frame.

        Returns
        -------
        xarray.DataArray
            The combined and sorted frequency axis.

        Raises
        ------
        AssertionError
            If the frequency reference frames are inconsistent across Measurement Sets.
        """
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
        """
        Determine the maximum dimension sizes across all Measurement Sets.

        This internal method iterates over all Measurement Sets, tracking the largest size for each dimension.
        The result is a dictionary mapping dimension names to their maximum sizes found across the Processing Set.

        Returns
        -------
        dict
            A dictionary with dimension names as keys and their maximum sizes as values.
        """
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
