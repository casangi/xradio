"""IO helpers for image test assets."""

from __future__ import annotations

import shutil
from pathlib import Path


def download_image(fname: str, directory: str | Path = ".") -> Path:
    """Download an image asset to disk without opening it.

    Mirrors the ``download_measurement_set`` signature so image test helpers
    follow the same convention as measurement-set helpers.

    Parameters
    ----------
    fname : str
        Name of the image asset to download (e.g. ``"casa_test_image.im"``).
    directory : str or Path, optional
        Target directory.  Defaults to the current working directory
        (``"."``).

    Returns
    -------
    Path
        Absolute path to the downloaded asset.
    """
    from toolviper.utils.data import download

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / fname
    if not dest.exists():
        download(file=fname, folder=str(directory))
        assert dest.exists(), f"Could not download {fname!r} into {directory}"
    return dest


def download_and_open_image(fname: str, directory: str | Path = "."):
    """Download an image asset and return it as an opened ``xr.Dataset``.

    Parameters
    ----------
    fname : str
        Name of the image asset to download.
    directory : str or Path, optional
        Target directory.  Defaults to the current working directory.

    Returns
    -------
    xr.Dataset
        The opened image dataset.
    """
    from xradio.image import open_image

    path = download_image(fname, directory=directory)
    return open_image(str(path))


def remove_path(path: str | Path) -> None:
    """Delete a file or directory tree if it exists.

    Parameters
    ----------
    path : str or Path
        Path to the file or directory to remove.  A no-op when the path
        does not exist.
    """
    path = Path(path)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(str(path))
        else:
            path.unlink()
