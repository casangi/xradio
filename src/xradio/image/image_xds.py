import weakref
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import xarray as xr

from xradio._utils.xarray_helpers import (
    create_new_data_group,
    delete_data_variables,
)

IMAGE_DATASET_TYPES = {"image_dataset"}


class InvalidAccessorLocation(ValueError):
    """
    Raised by ImageXds accessor functions called on a wrong Dataset (not image).
    """

    pass


class ImageXds:
    """Accessor to the Image Dataset.

    The accessor holds its Dataset through a WEAK reference: xarray caches the
    accessor instance on the Dataset (``ds._cache["xr_img"]``) on first
    access, so a strong back-reference would form a reference cycle that
    keeps the entire image Dataset (and its arrays) alive until a full
    garbage-collection pass. The 2026-08 Frontera memory diagnosis traced
    ~1.5 GB of cyclic garbage per imaging task to exactly this cycle. With a
    weak reference the Dataset dies deterministically by refcount; the
    accessor is only ever used as ``ds.xr_img.method()``, where ``ds`` itself
    keeps the Dataset alive for the duration of the call.
    """

    def __init__(self, dataset: xr.Dataset):
        """
        Initialize the ImageXds instance.

        Parameters
        ----------
        dataset: xarray.Dataset
            The image Dataset node to construct an ImageXds accessor.
        """

        self._xds_strong: xr.Dataset | None = dataset
        self._xds_ref: weakref.ref | None = None
        self.meta = {"summary": {}}

    @property
    def _xds(self) -> xr.Dataset:
        if self._xds_strong is not None:
            return self._xds_strong
        xds = self._xds_ref() if self._xds_ref is not None else None
        if xds is None:
            raise ReferenceError(
                "The Dataset behind this ImageXds accessor no longer exists. "
                "Access the accessor as ds.xr_img.<method>() rather than "
                "keeping the accessor object alive beyond its Dataset."
            )
        return xds

    @_xds.setter
    def _xds(self, dataset: xr.Dataset) -> None:
        # Preserve the current reference mode on rebinding (e.g. from
        # add_uv_coordinates): weak stays weak, strong stays strong.
        if self._xds_strong is None and self._xds_ref is not None:
            self._xds_ref = weakref.ref(dataset)
        else:
            self._xds_strong = dataset

    def _weaken(self) -> "ImageXds":
        """Switch to a WEAK back-reference; called by the accessor-protocol
        factory below. xarray caches accessor instances on the Dataset
        (``ds._cache["xr_img"]``), so a strong back-reference would form a
        reference cycle keeping the whole image Dataset alive until a full gc
        pass (the 2026-08 memory diagnosis: ~1.5 GB/task). Directly
        constructed instances keep their strong reference (wrapper
        semantics)."""
        if self._xds_strong is not None:
            self._xds_ref = weakref.ref(self._xds_strong)
            self._xds_strong = None
        return self

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

        self._xds.attrs["data_groups"][new_data_group_name] = new_data_group
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

        # Calculate uv coordinates in meters based on l and m. _make_uv_coords assumes reference pixel at center (not necessary the case).
        delta = self.get_lm_cell_size()
        image_size = [self._xds.sizes["l"], self._xds.sizes["m"]]

        u = self._xds.coords["l"].values / ((delta[0] ** 2) * image_size[0])
        v = self._xds.coords["m"].values / ((delta[1] ** 2) * image_size[1])

        # Keep a strong local reference: self._xds only holds the Dataset
        # weakly, so assigning and immediately re-reading it would let the
        # new Dataset be freed between the two statements.
        xds = self._xds.assign_coords({"u": u, "v": v})
        self._xds = xds
        return xds

    def get_uv_in_lambda(self, frequency: float):
        """Get the uv coordinates in wavelengths for a specific frequency from the image Dataset.

        Parameters
        ----------
        frequency : float
            The frequency in Hz to calculate the uv coordinates in wavelengths.

        Returns
        -------
        np.ndarray
            The uv coordinates in wavelengths.
        """

        #    if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
        #        raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")
        self.test_func()

        c = 299792458.0  # Speed of light in m/s
        wavelength = c / frequency  # Wavelength in meters

        u_in_lambda = self._xds.coords["u"] / wavelength
        v_in_lambda = self._xds.coords["v"] / wavelength

        return u_in_lambda, v_in_lambda

    def get_reference_pixel_indices(self):
        """Get the reference pixel indices from the image Dataset. The reference pixel is defined as the pixel where l=0 and m=0 or u=0 and v=0.

        Returns
        -------
        dict
            A dictionary with the reference pixel indices for each dimension.
        """

        #        if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
        #            raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")
        self.test_func()

        image_center_index = None

        if "l" in self._xds.coords:
            l_index = np.where(self._xds.coords["l"].values == 0)[0][0]
            m_index = np.where(self._xds.coords["m"].values == 0)[0][0]

            lm_indexes = np.array([l_index, m_index])
            image_center_index = lm_indexes
        else:
            lm_indexes = None

        if "u" in self._xds.coords:
            u_index = np.where(self._xds.coords["u"].values == 0)[0][0]
            v_index = np.where(self._xds.coords["v"].values == 0)[0][0]
            uv_indexes = np.array([u_index, v_index])

            assert np.array_equal(lm_indexes, uv_indexes), (
                "lm and uv reference pixel indices do not match."
            )
            image_center_index = uv_indexes
        else:
            uv_indexes = None

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
            data_group_name = indexers_kwargs["data_group_name"]
            del indexers_kwargs["data_group_name"]
        elif (indexers is not None) and ("data_group_name" in indexers):
            data_group_name = indexers["data_group_name"]
            del indexers["data_group_name"]
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

            sel_img_xds = self._xds

            sel_corr_xds = self._xds.sel(
                indexers, method, tolerance, drop, **indexers_kwargs
            ).drop_vars(data_variables_to_drop)

            sel_img_xds = sel_corr_xds

            sel_img_xds.attrs["data_groups"] = {
                data_group_name: self._xds.attrs["data_groups"][data_group_name]
            }

            return sel_img_xds
        else:
            return self._xds.sel(indexers, method, tolerance, drop, **indexers_kwargs)

    def delete_data_variables(self, variables: list[str]) -> xr.Dataset:
        """Delete data variables from the image dataset and all data groups.

        Parameters
        ----------
        variables : list of str
            List of data variable names to delete.

        Returns
        -------
        xarray.Dataset
            ImageXds Dataset with specified data variables deleted.
        """
        if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{getattr(self._xds, 'path', 'dataset')} is not a image node "
                f"(type {self._xds.attrs.get('type')})."
            )

        delete_data_variables(self._xds, variables)
        return self._xds


def _xr_img_accessor_factory(dataset: xr.Dataset) -> ImageXds:
    """Accessor-protocol factory: weak-referenced ImageXds (no cache cycle)."""
    return ImageXds(dataset)._weaken()


xr.register_dataset_accessor("xr_img")(_xr_img_accessor_factory)
