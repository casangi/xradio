from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import xarray as xr

from xradio._utils.xarray_helpers import (
    create_new_data_group,
    delete_data_variables,
    register_uncached_accessor,
    replace_data_groups,
)
from xradio.image._util.common import (
    _compute_sky_reference_pixel,
    _linear_axis_reference_pixel,
)

IMAGE_DATASET_TYPES = {"image_dataset"}

#: u/v ``units`` values meaning wavelengths (the image schema convention).
_WAVELENGTH_UNITS = {"lambda", "wavelength", "wavelengths"}

# A reference pixel this close to an integer (in pixels) is a pixel of the grid
_PIXEL_INDEX_TOLERANCE = 1e-9


def _pixel_indices(pixels: np.ndarray) -> np.ndarray:
    """Integer indices when every (fractional) pixel position is a pixel of
    the grid, else the positions themselves."""
    nearest = np.round(pixels)
    if np.all(np.abs(pixels - nearest) <= _PIXEL_INDEX_TOLERANCE):
        return nearest.astype(np.int64)
    return pixels


class InvalidAccessorLocation(ValueError):
    """
    Raised by ImageXds accessor functions called on a wrong Dataset (not image).
    """

    pass


class ImageXds:
    """Accessor to the Image Dataset.

    Registered as ``xr.Dataset.xr_img`` without xarray's accessor cache: every
    ``ds.xr_img`` builds a new accessor that holds ``ds`` strongly and is
    never stored on ``ds``. Chained calls on temporaries
    (``ds.isel(...).xr_img.get_lm_cell_size()``) therefore work, and there is
    no ``ds -> accessor -> ds`` reference cycle: a dataset dies by reference
    counting as soon as its last reference (including any accessor kept by
    the caller) is dropped. The 2026-08 Frontera memory diagnosis traced
    ~1.5 GB of cyclic garbage per imaging task to the cycle that xarray's
    cached accessors form with a strong back-reference.
    """

    def __init__(self, dataset: xr.Dataset):
        """
        Initialize the ImageXds instance.

        Parameters
        ----------
        dataset: xarray.Dataset
            The image Dataset node to construct an ImageXds accessor.
        """

        self._xds: xr.Dataset = dataset
        self.meta = {"summary": {}}

    def test_func(self):
        if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
            location = (
                self._xds.path if getattr(self._xds, "path", None) else "In-memory xds"
            )
            raise InvalidAccessorLocation(f"{location} is not of type image.")

        return "Hallo"

    def add_data_group(
        self,
        new_data_group_name: str,
        new_data_group: dict | None = None,
        data_group_dv_shared_with: str = None,
    ) -> xr.Dataset:
        """Adds a data group to the image Dataset, grouping the given data, weight, flag, etc. variables
        and field_and_source_xds.

        Parameters
        ----------
        new_data_group_name : str
            _description_
        new_data_group : dict
            _description_, by default Non
        data_group_dv_shared_with : str, optional
            _description_, by default "base"

        Returns
        -------
        xr.Dataset
          Image Dataset with the new group added
        """

        #    if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
        #        raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")

        self.test_func()

        if new_data_group is None:
            new_data_group = {}

        new_data_group_name, new_data_group = create_new_data_group(
            self._xds,
            "image",
            new_data_group_name,
            new_data_group,
            data_group_dv_shared_with=data_group_dv_shared_with,
        )

        # Replace the attrs mapping: the data_groups dict may be shared with
        # the datasets this one was derived from (or derived into).
        replace_data_groups(
            self._xds,
            {**self._xds.attrs["data_groups"], new_data_group_name: new_data_group},
        )
        return self._xds

    def get_lm_cell_size(self):
        """Get the lm cell size in radians from the image Dataset.

        Returns
        -------
        float
            The lm cell size in radians.
        """

        #    if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
        #        raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")
        self.test_func()

        l_cell_size = self._xds.coords["l"][1].values - self._xds.coords["l"][0].values
        m_cell_size = self._xds.coords["m"][1].values - self._xds.coords["m"][0].values

        return np.array([l_cell_size, m_cell_size])

    def add_uv_coordinates(self) -> xr.Dataset:
        """Adds the uv coordinates in wavelengths to the image Dataset.

        The ``u`` and ``v`` coordinates are the aperture plane coordinates
        conjugate to ``l`` and ``m``, in wavelengths (``units`` ``"lambda"``,
        the image schema convention), so they do not depend on frequency.

        Parameters
        ----------

        Returns
        -------
        xr.Dataset
            Image Dataset with the uv coordinates added.
        """

        #    if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
        #        raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")
        self.test_func()

        # self._xds = _make_uv_coords(self._xds,image_size=image_size, sky_image_cell_size=self.get_lm_cell_size())

        # Calculate uv coordinates in wavelengths based on l and m: the uv
        # cell size is 1 / (image size * lm cell size). _make_uv_coords assumes reference pixel at center (not necessary the case).
        delta = self.get_lm_cell_size()
        image_size = [self._xds.sizes["l"], self._xds.sizes["m"]]

        u = self._xds.coords["l"].values / ((delta[0] ** 2) * image_size[0])
        v = self._xds.coords["m"].values / ((delta[1] ** 2) * image_size[1])

        xds = self._xds.assign_coords(
            {"u": ("u", u, {"units": "lambda"}), "v": ("v", v, {"units": "lambda"})}
        )
        self._xds = xds
        return xds

    def get_uv_in_lambda(self, frequency: float):
        """Get the uv coordinates in wavelengths for a specific frequency from the image Dataset.

        The ``units`` attribute of the ``u`` and ``v`` coordinates decides the
        conversion: coordinates already in wavelengths (``"lambda"`` or
        ``"wavelengths"``, the convention of the image schema, the readers,
        the ``make_empty_*`` factories and :meth:`add_uv_coordinates`) are
        returned unchanged, and coordinates in a length unit (e.g. ``"m"``)
        are divided by the wavelength ``c / frequency``.

        Parameters
        ----------
        frequency : float
            The frequency in Hz to calculate the uv coordinates in wavelengths.
            Not used when the coordinates are already in wavelengths.

        Returns
        -------
        tuple of xarray.DataArray
            The u and v coordinates in wavelengths.

        Raises
        ------
        ValueError
            If a coordinate has no ``units`` or units that are neither
            wavelengths nor a length.
        """

        #    if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
        #        raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")
        self.test_func()

        c = 299792458.0  # Speed of light in m/s
        wavelength = c / frequency  # Wavelength in meters

        u_in_lambda, v_in_lambda = (
            _uv_in_wavelengths(self._xds.coords[name], wavelength)
            for name in ("u", "v")
        )

        return u_in_lambda, v_in_lambda

    def get_reference_pixel_indices(self):
        """Get the reference pixel indices from the image Dataset. The reference pixel is defined as the pixel where l=0 and m=0 or u=0 and v=0.

        The reference pixel does not have to be a pixel of the image: a
        cutout that does not contain the reference direction has its
        reference pixel outside the image.

        Returns
        -------
        numpy.ndarray
            The (l, m) or (u, v) indices of the reference pixel: integers
            when it is a pixel of the image, else its fractional (float)
            pixel position, extrapolated with the coordinate increment
            outside the image (negative, or beyond the last pixel).
        """

        #        if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
        #            raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")
        self.test_func()

        lm_indexes = None
        if "l" in self._xds.coords:
            lm_indexes = _pixel_indices(_compute_sky_reference_pixel(self._xds))

        uv_indexes = None
        if "u" in self._xds.coords:
            uv_indexes = _pixel_indices(
                np.array(
                    [
                        _linear_axis_reference_pixel(self._xds.coords[name].values)
                        for name in ("u", "v")
                    ]
                )
            )
            if lm_indexes is not None:
                assert np.array_equal(lm_indexes, uv_indexes), (
                    "lm and uv reference pixel indices do not match."
                )

        image_center_index = uv_indexes if uv_indexes is not None else lm_indexes
        if image_center_index is None:
            raise ValueError("No lm or uv coordinates found in the image Dataset.")

        return image_center_index

    def sel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        method: str | None = None,
        tolerance: int | float | Iterable[int | float] | None = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> xr.Dataset:
        """
        Select data along dimension(s) by label. Alternative to `xarray.Dataset.sel <https://xarray.pydata.org/en/stable/generated/xarray.Dataset.sel.html>`__ so that a data group can be selected by name by using the `data_group_name` parameter.
        For more information on data groups see `Data Groups <https://xradio.readthedocs.io/en/latest/measurement_set_overview.html#Data-Groups>`__ section. See `xarray.Dataset.sel <https://xarray.pydata.org/en/stable/generated/xarray.Dataset.sel.html>`__ for parameter descriptions.

        Returns
        -------
        xarray.Dataset
            xarray Dataset with ImageXds accessors

        Examples
        --------
        >>> # Select data group 'robust0.5' and polarization 'XX'.
        >>> selected_img_xds = img_xds.xr_img.sel(data_group_name='robust0.5', polarization='XX')
        """

        #        if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
        #            raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")
        self.test_func()

        if "data_group_name" in indexers_kwargs:
            data_group_name = indexers_kwargs.pop("data_group_name")
        elif (indexers is not None) and ("data_group_name" in indexers):
            # Copy rather than edit the caller's indexers mapping
            indexers = dict(indexers)
            data_group_name = indexers.pop("data_group_name")
        else:
            data_group_name = None

        if data_group_name is not None:
            sel_data_group_set = set(
                self._xds.attrs["data_groups"][data_group_name].values()
            ) - {"date", "description"}

            data_variables_to_drop = []
            for dg in self._xds.attrs["data_groups"].values():
                dg_copy = dg.copy()
                dg_copy.pop("date", None)
                dg_copy.pop("description", None)
                temp_set = set(dg_copy.values()) - sel_data_group_set
                data_variables_to_drop.extend(list(temp_set))

            data_variables_to_drop = list(set(data_variables_to_drop))

            sel_img_xds = self._xds.sel(
                indexers, method, tolerance, drop, **indexers_kwargs
            ).drop_vars(data_variables_to_drop)

            # Replace the attrs mapping and copy the selected group: both are
            # otherwise shared with self._xds (sel shares the attrs dict).
            replace_data_groups(
                sel_img_xds,
                {
                    data_group_name: dict(
                        self._xds.attrs["data_groups"][data_group_name]
                    )
                },
            )

            return sel_img_xds
        else:
            return self._xds.sel(indexers, method, tolerance, drop, **indexers_kwargs)

    def delete_data_variables(self, variables: list[str]) -> xr.Dataset:
        """Delete data variables from the image dataset and all data groups.

        The variables are deleted from this dataset in place, and every data
        group role that refers to one of them is removed. Datasets that share
        data with this one (e.g. made with ``isel``, ``sel``, ``copy`` or
        :meth:`sel`) keep their variables and data groups.

        Parameters
        ----------
        variables : list of str
            List of data variable names to delete.

        Returns
        -------
        xarray.Dataset
            ImageXds Dataset with specified data variables deleted.

        Raises
        ------
        ValueError
            If a name is not a data variable of the dataset (nothing is
            deleted then).
        """
        if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{getattr(self._xds, 'path', 'dataset')} is not a image node "
                f"(type {self._xds.attrs.get('type')})."
            )

        delete_data_variables(self._xds, variables)
        return self._xds


def _uv_units(coord: xr.DataArray):
    """Return the units of a u or v coordinate, or None.

    The units are the ``units`` attribute; the ``make_empty_*`` factories
    instead store the coordinate attrs as a quantity dict, with the units in
    ``attrs["attrs"]["units"]``.
    """
    units = coord.attrs.get("units")
    if units is None:
        nested = coord.attrs.get("attrs")
        if isinstance(nested, Mapping):
            units = nested.get("units")
    if isinstance(units, list | tuple) and len(units) == 1:
        units = units[0]
    return units


def _uv_in_wavelengths(coord: xr.DataArray, wavelength: float) -> xr.DataArray:
    """Return a u or v coordinate in wavelengths, given the wavelength in m."""
    units = _uv_units(coord)
    if units is None:
        raise ValueError(
            f"The {coord.name} coordinate has no units, so it cannot be converted "
            "to wavelengths. Set its 'units' attribute to 'lambda' (wavelengths, "
            "the image schema convention) or to a length unit such as 'm'."
        )
    if str(units).strip().lower() in _WAVELENGTH_UNITS:
        return coord.copy()

    from astropy import units as u

    try:
        to_meters = u.Unit(units).to(u.m)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot convert the {coord.name} coordinate with units {units!r} to "
            "wavelengths: the units must be 'lambda' (wavelengths) or a length "
            "such as 'm'."
        ) from exc
    converted = coord * (to_meters / wavelength)
    converted.attrs = {"units": "lambda"}
    return converted


register_uncached_accessor("xr_img", xr.Dataset)(ImageXds)
