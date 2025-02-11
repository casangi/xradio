from __future__ import annotations

from collections import defaultdict
import importlib
import importlib.metadata
import os.path
from packaging.version import Version
import shutil
from typing import Dict, List, Literal, Set, Tuple
import warnings

import zarr

# Files that are associated with zarr versions
ZARR_FILE_VERSION: Dict[str, int] = {
  ".zgroup": 2,
  "zarr.json": 3,
}

# https://confluence.skatelescope.org/display/SEC/Datatree+proposal
VALID_OPTIONS: Set[float] = {
  1.0,
  2.0,
  2.5,
  3.0
}

class DatatreeBuilder:
  """Builds a Datatree representation from an existing Processing Set representation"""

  """Processing Set url"""
  _url: str
  """Datatree proposal option https://confluence.skatelescope.org/display/SEC/Datatree+proposal"""
  _option: float
  """Datatree url"""
  _destination_url: str
  """Configures overwiting of existing data at the Datatree url"""
  _overwrite: bool
  """Configures copying or moving data from the Processing Set"""
  _copy: bool
  """Configures removal of the xds suffix from Processing Set datasets"""
  _remove_suffix: bool
  """Configures the location of the zarr metadata consolidation at either
  the Datatree root or the at the partition level"""
  _consolidate_at: Literal["root", "partition"]

  def __init__(self):
    self._url = ""
    self._copy = True
    self._destination_url = ""
    self._overwrite = True
    self._option = 2.0
    self._consolidate_at = "partition"
    self._remove_suffix = False

  def with_url(self, url: str) -> DatatreeBuilder:
    """ Sets the input url """
    self._url = url.rstrip(os.path.sep)
    return self

  def with_copy(self) -> DatatreeBuilder:
    """ Configures copying of data from the Processing Set to the DataTree.
    If False, data will be moved """
    self._copy = True
    return self

  def with_move(self) -> DatatreeBuilder:
    """ Configures moving of Processing Set data to the DataTree"""
    self._copy = False
    return self

  def with_remove_suffix(self) -> DatatreeBuilder:
    """ Configures removal of xds suffix from dataset name """
    self._remove_suffix = True
    return self

  def with_option(self, option: float = 2.5) -> DatatreeBuilder:
    self._option = option
    return self

  def with_destination(self, destination_url: str) -> DatatreeBuilder:
    """ Sets the destination url of the DataTree """
    self._destination_url = destination_url
    return self

  def with_overwrite(self, overwrite: bool = True) -> DatatreeBuilder:
    """ Configures overwriting any existing data in the destination url"""
    self._overwrite = overwrite
    return self

  def with_consolidate_at(self, at: str) -> DatatreeBuilder:
    assert at in {"root", "partition"}
    self._consolidate_at = at
    return self

  def maybe_generate_destination_url(self) -> str:
    """ Returns destination url if populated, else generates one from the input url"""
    if self._destination_url:
      if not self._destination_url.endswith(".zarr"):
        warnings.warn(
          f"{self._destination_url} does not have a .zarr extension "
          f"and will not be automatically recognised by xarray as a zarr store.\n"
          f"The engine will need to be explicitly specified via "
          f"xarray.open_datatree({self._destination_url}, engine='zarr')")

      return self._destination_url

    root, base_url = os.path.split(self._url)
    # The xarray zarr engine is engaged if a .zarr extension is encountered
    # https://github.com/pydata/xarray/blob/1189240b2631fa27dec0cbea76bf3cf977b42fce/xarray/backends/zarr.py#L1527-L1535
    # Strip off non standard extensions and replace with .zarr
    name, _ = os.path.splitext(base_url)
    return os.path.join(root, f"{name}_datatree.zarr")

  def maybe_rename_dataset(self, name: str) -> str:
    """ Returns a possibly renamed dataset name if xds suffix removal is configured,
     otherwise the original name """
    return name.replace("_xds", "") if self._remove_suffix else name

  @property
  def strategy_string(self) -> str:
    """ Returns a human readable string describing the Datatree build strategy"""
    bits = ["Datatree Build Strategy"]
    tab = " " * 2
    bits.append(f"{tab}* Datatree proposal option:'{self._option}'")
    bits.append(f"{tab}* Data in '{self._url}' will be {'copied' if self._copy else 'moved'} to")
    bits.append(f"{tab}  '{self.maybe_generate_destination_url()}'")
    if self._overwrite:
      bits.append(f"{tab}* '{self.maybe_generate_destination_url()}' will be overwritten")

    if self._consolidate_at == "root":
      bits.append(f"{tab}* Metadata will be consolidated at the Datatree root")
    elif self._consolidate_at == "partition":
      bits.append(f"{tab}* Metadata will be consolidated at the Partition level, below the DataTree root")

    if self._remove_suffix:
      bits.append(f"{tab}* 'xds' will be removed from dataset names")

    return "\n".join(bits)

  def build(self):
    """ Build the Datatree from the configured input """
    if self._option not in VALID_OPTIONS:
      raise ValueError(f"{self._option} is a valid datatree struture option {VALID_OPTIONS}")

    if not self._url or not os.path.exists(self._url) or not os.path.isdir(self._url):
      raise FileNotFoundError(f"{self._url} does not exist or is not a directory")

    # Find non-hidden directories
    with os.scandir(self._url) as it:
      partitions = [e.name for e in it if e.is_dir() and not e.name.startswith(".")]

    if len(partitions) == 0:
      raise FileNotFoundError(f"{self._url} does not contain any partitions")

    # Iterate through the partitions, finding datasets and
    # inferring their associated zarr versions
    partition_datasets: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    for partition in partitions:
      partition_dir = os.path.join(self._url, partition)
      with os.scandir(partition_dir) as it:
        for dataset_dir in [e.name for e in it if e.is_dir()]:
          with os.scandir(os.path.join(partition_dir, dataset_dir)) as it:
            zarr_files = {e.name for e in it if e.is_file() and e.name in ZARR_FILE_VERSION}
            zarr_versions = {ZARR_FILE_VERSION[f] for f in zarr_files}

            if len(zarr_versions) == 0:
              continue
            elif len(zarr_versions) > 1:
              raise ValueError(
                f"Multiple zarr versions {zarr_versions} "
                f"found in {os.path.join(partition_dir, dataset_dir)}"
              )
            elif next(iter(zarr_versions)) != 2:
              raise NotImplementedError(f"Conversion of Zarr v{zarr_version} Datasets")

            partition_datasets[partition].append((dataset_dir, next(iter(zarr_versions))))

    zarr_library_version = Version(importlib.metadata.version("zarr"))

    if zarr_library_version >= Version("3.0.0"):
      raise ValueError(f"zarr version {zarr_library_version} >= 3.0.0")

    # Create root zarr group
    # This code is probably specific to zarr versions < 3.
    # For zarr >= 3, zarr.create_group looks to be appropriate.
    destination_root = self.maybe_generate_destination_url()

    if self._overwrite:
      shutil.rmtree(destination_root, ignore_errors=True)
      os.makedirs(destination_root, exist_ok=False)
      root = zarr.open_group(store=destination_root, mode="w")
    else:
      root = zarr.open_group(store=destination_root, mode="w-")

    # Create zarr partition groups below the root
    for p, (partition, datasets) in enumerate(partition_datasets.items(), 1):
      # Create root children
      if self._option in {1.0, 2.0, 2.5}:
        # Partition is a child of the tree root
        root.create_group(partition)
      elif self._option == 3.0 and p == 1:
        # Partition is a child of the table type, which
        # in turn is a child of the tree root
        # Assumption: Each partition has the same datasets.
        # Good enough for a prototype
        for dataset, _ in datasets:
          root.create_group(self.maybe_rename_dataset(dataset))

      # Copy/move datasets into the appropriate locations
      for dataset, zarr_version in datasets:
        source = os.path.join(self._url, partition, dataset)
        dest_dataset = self.maybe_rename_dataset(dataset)

        if self._option == 1.0:
          dest = os.path.join(destination_root, partition, dest_dataset)
        elif self._option == 2.0:
          if dataset == "correlated_xds":
            # Visibility dataset forms the root of the partition
            dest = os.path.join(destination_root, partition)
          else:
            # Other datasets are children of the visibility dataset
            dest = os.path.join(destination_root, partition, dest_dataset)
        elif self._option == 2.5:
          if dataset == "correlated_xds":
            # Visibility dataset forms the root of the partition
            dest = os.path.join(destination_root, partition)
          elif dataset == "antenna_xds":
            # The antenna dataset is a child of the tree root
            # For prototyping purposes, we only write it once
            dest = os.path.join(destination_root, dest_dataset)
            if os.path.exists(dest):
              continue
          else:
            # Other datasets are children of the visibility dataset
            dest = os.path.join(destination_root, partition, dest_dataset)
        elif self._option == 3.0:
          dest = os.path.join(destination_root, dest_dataset, f"xds{p}")
        else:
          raise NotImplementedError(f"Datatree proposal option {self._option}")

        # Perform the copy or move
        if self._copy:
          shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
          shutil.rmtree(dest, ignore_errors=True)
          shutil.move(source, dest)

      # Consolidate metadata at the partition level
      if self._consolidate_at == "partition":
        if self._option in {1.0, 2.0, 2.5}:
          zarr.consolidate_metadata(os.path.join(destination_root, partition))
        elif self._option == 3.0 and p == len(partition_datasets):
          # Consolidate after the last partition has been created
          for dataset, _ in datasets:
            dest_dataset = self.maybe_rename_dataset(dataset)
            zarr.consolidate_metadata(os.path.join(destination_root, dest_dataset))

    if self._consolidate_at == "root":
      zarr.consolidate_metadata(destination_root)
