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
