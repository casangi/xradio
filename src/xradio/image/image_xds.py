from collections.abc import Mapping, Iterable
import datetime
from typing import Any, Union

import numpy as np
import xarray as xr

from xradio._utils.list_and_array import to_list

IMAGE_DATASET_TYPES = {"image"}


class InvalidAccessorLocation(ValueError):
    """
    Raised by ImageXds accessor functions called on a wrong Dataset (not image).
    """

    pass


@xr.register_dataset_accessor("xr_img")
class ImageXds:
    """Accessor to the Image Dataset. Provides image specific functionality
    such as:

        - get_partition_info(): produce an info dict with a general image description including
          intents, SPW name, field and source names, etc.
        - get_field_and_source_xds() to retrieve the field_and_source_xds for a given data
          group.
        - sel(): select data by dimension labels, for example by data group and polaritzation

    """

    _xds: xr.Dataset

    def __init__(self, dataset: xr.Dataset):
        """
        Initialize the ImageXds instance.

        Parameters
        ----------
        dataset: xarray.Dataset
            The image Dataset node to construct an ImageXds accessor.
        """

        self._xds = dataset
        self.meta = {"summary": {}}

    def test_func(self):
        if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
            raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")

        return "Hallo"
