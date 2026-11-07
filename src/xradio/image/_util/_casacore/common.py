from collections.abc import Generator
from contextlib import contextmanager

from xradio._utils._casacore.backend import images


@contextmanager
def _open_image_ro(infile: str) -> Generator[images.image, None, None]:
    image = images.image(infile)
    try:
        yield image
    finally:
        # there is no obvious way to close a python-casacore image, so
        # just delete the object to clear it from the table cache
        # image.unlock() # not necessary
        del image


@contextmanager
def _create_new_image(
    outfile: str, shape: list[int], mask="", value="default"
) -> Generator[images.image, None, None]:
    # new image will be opened rw
    # the crux of the issue here seems to be that python has no single
    # precision floating point value, but we need single precision for
    # most images. The image constructor gets that right if values isn't
    # suppliked. Hence two calls with values not present and values present
    if value == "default":
        image = images.image(outfile, maskname=mask, shape=shape)
    else:
        image = images.image(outfile, maskname=mask, shape=shape, values=value)
    try:
        yield image
    finally:
        # Explicitly calling unlock() is important for downstream parallel reads across multiple processes.
        # Without it, Dask workers may mistakenly consider a freshly written image from another process
        # as invalid—even if the image directory appears to be fully written and present.
        image.unlock()
        del image


_image_flag = "flag"
_beam_fit_params = "beam_fit_params"
# _native_types = ["FREQ", "VRAD", "VOPT", "BETA", "WAVE", "AWAV"]
_object_name = "object_name"
_pointing_center = "pointing_center"


def _get_casa_image_tile_shape(
    img_full_path: str, colname: str | None = None
) -> list[int] | None:
    """Retrieve the storage manager tile shape (in Fortran/casacore axis order)
    for a CASA image column from its table dminfo.
    """
    from xradio._utils._casacore.tables import open_table_ro

    try:
        with open_table_ro(img_full_path) as tb:
            dminfo = tb.getdminfo()
            target_groups = []
            if colname:
                for group in dminfo.values():
                    if colname in group.get("COLUMNS", []):
                        target_groups.append(group)
            if not target_groups:
                target_groups = list(dminfo.values())
            for group in target_groups:
                spec = group.get("SPEC", {})
                for hc in spec.get("HYPERCUBES", {}).values():
                    if "TileShape" in hc and len(hc["TileShape"]) > 0:
                        return [int(s) for s in hc["TileShape"]]
                if "DEFAULTTILESHAPE" in spec and len(spec["DEFAULTTILESHAPE"]) > 0:
                    return [int(s) for s in spec["DEFAULTTILESHAPE"]]
    except Exception:
        pass
    return None
