import argparse
import contextlib
import logging

from xradio.datatree.datatree_builder import DatatreeBuilder
import xradio.datatree.datatree_accessor  # noqa
from xradio.datatree.datatree_accessor import InvalidAccessorLocation, VISIBILITY_DATASET_TYPES, DATASET_TYPES


@contextlib.contextmanager
def assert_raises(expected_exception):
    exception_caught = None
    try:
        yield  # Code within the 'with' block executes here
    except Exception as exc:
        exception_caught = exc
        if not isinstance(exc, expected_exception):
            raise AssertionError(
                f"Expected exception {expected_exception.__name__}, but got {type(exc).__name__}"
            )
    else:  # No exception raised
        raise AssertionError(f"Expected exception {expected_exception.__name__} was not raised")



PROPOSAL_URL = "https://confluence.skatelescope.org/display/SEC/Datatree+proposal"

def create_parser():
  p = argparse.ArgumentParser()
  p.add_argument("ps", help="Processing Set")
  p.add_argument("-m", "--move",
                 help="Moves Processing Set Datasets instead of copying them")
  p.add_argument("-o", "--option",
                 help=f"Datatree proposal option as described at {PROPOSAL_URL}",
                 default=1.0,
                 type=float)
  p.add_argument("-rs", "--remove-suffix",
                 help="Removes xds from dataset name",
                 action="store_true")
  p.add_argument("-c", "--consolidate-at",
                 default="root",
                 choices=["root", "partition"])
  p.add_argument("-ol", "--output",
                 help=(
                      "Output location. If empty, will be generated "
                      "from the processing set name."
                 ),
                 default="")
  p.add_argument("--test", choices=["none", "open", "read"], default="open")
  return p

if __name__ == "__main__":
  args = create_parser().parse_args()
  logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
  builder = DatatreeBuilder(args.ps)
  builder = builder.with_option(args.option)
  builder = builder.with_consolidate_at(args.consolidate_at)
  builder = builder.with_destination(args.output)
  builder = builder.with_overwrite(True)
  if args.move:
    builder = builder.with_move()
  if args.remove_suffix:
    builder = builder.with_remove_suffix()

  print(builder.strategy_string)
  builder.build()

  print(f"dt = xarray.open_datatree(\"{builder.maybe_generate_destination_url()}\", engine=\"zarr\")")

  if args.test in {"open", "read"}:
    import xarray
    from xarray import DataTree

    dt = xarray.open_datatree(builder.maybe_generate_destination_url())

    def pass_filter(node: DataTree) -> bool:
      return (
        node.attrs.get("type") in DATASET_TYPES
        # Propagates prototyping attributes required by the accessor
        or node.is_root
      )

    # Perform basic filtering and accessor testing
    for node in dt.filter(pass_filter).subtree:
      if node.attrs.get("type") in VISIBILITY_DATASET_TYPES:
        assert node.msa.antennas.attrs["type"] == "antenna"
        if weather := node.msa.weather:
           assert weather.attrs["type"] == "weather"
        if gain_curve := node.msa.gain_curve:
           assert gain_curve.attrs["type"] == "gain_curve"
        if field_and_source := node.msa.field_and_source():
           assert field_and_source.attrs["type"] == "field_and_source"
      else:
        with assert_raises(InvalidAccessorLocation):
          node.msa.weather

        with assert_raises(InvalidAccessorLocation):
          node.msa.antennas

    if args.test == "read":

      dt.load()


