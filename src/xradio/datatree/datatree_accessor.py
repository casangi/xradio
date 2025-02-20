import xarray
from xarray import DataTree

from xradio.datatree.datatree_builder import VALID_OPTIONS

class DatasetNotFound(ValueError):
  pass

class InvalidAccessorLocation(ValueError):
  pass


VISIBILITY_DATASET_TYPES = {"visibility", "spectrum", "wvr"}
SECONDARY_DATASET_TYPES = {"antenna", "field_and_source", "gain_curve", "system_calibration", "weather"}
DATASET_TYPES = VISIBILITY_DATASET_TYPES | SECONDARY_DATASET_TYPES


@xarray.register_datatree_accessor("msa")
class MeasurementSetAccessor:
  _dt: DataTree
  _option: float
  _remove_suffix: bool

  def __init__(self, datatree: DataTree):
    self._dt = datatree
    root = self._dt.root

    # option and remove_suffix would be unnecessary in a complete solution
    # but are used here to delegate based on the different prototype options
    # and configurations
    try:
      option = float(root.attrs["__datatree_proposal_option__"])
    except (KeyError, ValueError):
      option = 1.0

    if option not in VALID_OPTIONS:
      raise ValueError(f"{option} is not a valid DataTree Proposal option {VALID_OPTIONS}")

    self._option = option

    try:
      remove_suffix = bool(root.attrs["__datatree_proposal_remove_suffix__"])
    except (KeyError, ValueError):
      remove_suffix = False

    self._remove_suffix = remove_suffix

  def _maybe_rename_dataset(self, dataset: str) -> str:
    return dataset.replace("_xds", "") if self._remove_suffix else dataset

  @property
  def option(self) -> float:
    """ DataTree proposal option getter """
    return self._option

  @option.setter
  def option(self, value: float) -> None:
    """ DataTree proposal option setter """
    if value not in VALID_OPTIONS:
      raise ValueError(f"{value} is not a valid DataTree Proposal option {VALID_OPTIONS}")

    self._option = value

  def _get_optional_dataset(self, dataset: str) -> DataTree | None:
    name = self._maybe_rename_dataset(dataset)

    if self._dt.attrs.get("type") not in {"visibility", "spectrum", "wvr"}:
      raise InvalidAccessorLocation(
        f"{self._dt.path} is not a visibility node. "
        f"There is no {name} dataset associated with it.")

    try:
      if self._option == 1.0:
        other_ds = self._dt.siblings[name]
      elif self._option == 2.0:
        other_ds = self._dt.children[name]
      elif self._option == 2.5:
        other_ds = self._dt.children[name]
      elif self._option == 3.0:
        partition_str = self._dt.path.split("/")[-1]
        partition = int(partition_str[len("xds"):])
        other_ds = self._dt.root[f"/{name}/xds{partition}"]
      else:
        raise ValueError(f"Invalid option {self._option}")
    except KeyError as e:
      return None

    return other_ds

  def field_and_source(self, data_group_name="base") -> DataTree | None:
    return self._get_optional_dataset(f"field_and_source_xds_{data_group_name}")

  @property
  def gain_curve(self) -> DataTree | None:
    return self._get_optional_dataset("gain_curve_xds")

  @property
  def phase_calibration(self) -> DataTree | None:
    return self._get_optional_dataset("phase_calibration_xds")

  @property
  def system_calibration(self) -> DataTree | None:
    return self._get_optional_dataset("system_calibration_xds")

  @property
  def weather(self) -> DataTree | None:
    return self._get_optional_dataset("weather_xds")

  @property
  def antennas(self) -> DataTree:
    """ Access the antenna dataset """
    if self._dt.attrs.get("type") not in VISIBILITY_DATASET_TYPES:
      raise InvalidAccessorLocation(
        f"{self._dt.path} is not a visibility node. "
        f"There is no antenna dataset connected with it.")

    name = self._maybe_rename_dataset("antenna_xds")

    try:
      if self._option == 1.0:
        antenna_ds = self._dt.siblings[name]
      elif self._option == 2.0:
        antenna_ds = self._dt.children[name]
      elif self._option == 2.5:
        antenna_ds = self._dt.root[name]
      elif self._option == 3.0:
        partition_str = self._dt.path.split("/")[-1]
        partition = int(partition_str[len("xds"):])
        antenna_ds = self._dt.root[f"/{name}/xds{partition}"]
      else:
        raise ValueError(f"Invalid option {self._option}")
    except KeyError as e:
      raise DatasetNotFound(f"No antenna dataset found relative to {self._dt.path}") from e

    return antenna_ds

