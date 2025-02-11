import argparse
import logging

from xradio.datatree.datatree_builder import DatatreeBuilder

PROPOSAL_URL = "https://confluence.skatelescope.org/display/SEC/Datatree+proposal"

def create_parser():
  p = argparse.ArgumentParser()
  p.add_argument("ps", help="Processing Set")
  p.add_argument("-m", "--move")
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
  logging.basicConfig(format="%(levelname)s %(message)s", level=logging.DEBUG)
  builder = DatatreeBuilder()
  builder = builder.with_url(args.ps)
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
    dt = xarray.open_datatree(builder.maybe_generate_destination_url())
    if args.test == "read":
      dt.load()
