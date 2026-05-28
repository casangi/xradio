"""Unit tests for xradio.testing.image.io."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import sentinel

import pytest

from xradio.testing.image.io import download_and_open_image, download_image, remove_path

# --------------------------------------------------------------------------- #
# TestRemovePath                                                               #
# --------------------------------------------------------------------------- #


class TestRemovePath:
    """Tests for ``remove_path``."""

    def test_removes_existing_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        assert f.exists()
        remove_path(f)
        assert not f.exists()

    def test_removes_existing_directory(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        (d / "inner.txt").write_text("data")
        assert d.exists()
        remove_path(d)
        assert not d.exists()

    def test_noop_when_absent(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        remove_path(missing)  # must not raise


# --------------------------------------------------------------------------- #
# TestDownloadImage                                                            #
# --------------------------------------------------------------------------- #


class TestDownloadImage:
    """Tests for ``download_image``."""

    def test_returns_path_to_file(self, tmp_path, monkeypatch):
        fname = "test_asset.im"

        def fake_download(file, folder):
            (Path(folder) / file).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr("toolviper.utils.data.download", fake_download)
        result = download_image(fname, directory=tmp_path)
        assert result == tmp_path / fname
        assert result.exists()

    def test_raises_when_file_not_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr("toolviper.utils.data.download", lambda file, folder: None)
        with pytest.raises(FileNotFoundError):
            download_image("missing.im", directory=tmp_path)

    def test_skips_download_if_already_exists(self, tmp_path, monkeypatch):
        fname = "existing.im"
        (tmp_path / fname).mkdir()

        called = []
        monkeypatch.setattr(
            "toolviper.utils.data.download",
            lambda file, folder: called.append(True),
        )
        download_image(fname, directory=tmp_path)
        assert called == [], "download should not be called when file already exists"

    def test_creates_directory_if_missing(self, tmp_path, monkeypatch):
        nested = tmp_path / "a" / "b" / "c"
        fname = "asset.zarr"

        def fake_download(file, folder):
            (Path(folder) / file).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr("toolviper.utils.data.download", fake_download)
        download_image(fname, directory=nested)
        assert (nested / fname).exists()


# --------------------------------------------------------------------------- #
# TestDownloadAndOpenImage                                                     #
# --------------------------------------------------------------------------- #


class TestDownloadAndOpenImage:
    """Tests for ``download_and_open_image``."""

    def test_returns_opened_dataset(self, tmp_path, monkeypatch):
        fake_path = tmp_path / "img.im"
        fake_path.mkdir()
        fake_ds = sentinel.fake_dataset

        monkeypatch.setattr(
            "xradio.testing.image.io.download_image",
            lambda fname, directory=".": fake_path,
        )
        monkeypatch.setattr(
            "xradio.image.open_image",
            lambda path: fake_ds,
        )

        result = download_and_open_image("img.im", directory=tmp_path)
        assert result is fake_ds

    def test_passes_directory_to_download_image(self, tmp_path, monkeypatch):
        fake_path = tmp_path / "img.im"
        received = {}

        def fake_download(fname, directory="."):
            received["directory"] = directory
            return fake_path

        monkeypatch.setattr("xradio.testing.image.io.download_image", fake_download)
        monkeypatch.setattr("xradio.image.open_image", lambda path: sentinel.ds)

        download_and_open_image("img.im", directory=tmp_path)
        assert received["directory"] == tmp_path
