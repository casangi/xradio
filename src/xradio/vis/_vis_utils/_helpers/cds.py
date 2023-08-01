#  CASA Next Generation Infrastructure
#  Copyright (C) 2021, 2023 AUI, Inc. Washington DC, USA
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Tuple

import numpy as np
import xarray as xr

PartitionKey = Tuple[Any, ...]
# for intent-ddi partitioning
PartitionKeyAsNT = NamedTuple(
    "PartitionKey", [("spw_id", np.int32), ("pol_setup_id", np.int32), ("intent", str)]
)
VisSetPartitions = Dict[PartitionKey, xr.Dataset]
VisSetMetainfo = Dict[str, xr.Dataset]


@dataclass(frozen=True)
class CASAVisSet:
    metainfo: VisSetMetainfo
    partitions: VisSetPartitions
    descr: str

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        """
        Concise repr for CASA vis sets
        """
        class_name = type(self).__name__
        return (
            f"({class_name}\n metainfo (keys)={self.metainfo.keys()}"
            f"\n partitions (keys)={self.partitions.keys()}\n descr={self.descr!r})"
        )

    def _repr_html_(self):
        class_name = type(self).__name__
        return (
            f"({class_name}<p> <b>metainfo</b> (keys)={self.metainfo.keys()}</p><p>"
            f"<b>partitions</b> (keys)={self.partitions.keys()}</p><p> <b>descr</b>={self.descr!r})</p>"
        )
