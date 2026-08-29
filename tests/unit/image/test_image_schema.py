"""Unit tests for the image schema (``xradio.image.schema``).

* ``TestImageSchemaSynthetic``   → schema checking of synthetic image datasets
                                   (no downloads), including negative cases
* ``TestImageSchemaConstructors`` → schema classes used as constructors
* ``TestImageSchemaFromFormats`` → schema checking of images opened from CASA,
                                   FITS and zarr stores (downloads test data)

The dask cluster fixture is provided by ``conftest.py`` in this directory.
"""

from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from xradio.image import (
    make_empty_aperture_image,
    make_empty_lmuv_image,
    make_empty_sky_image,
    open_image,
    write_image,
)
from xradio.image.schema import (
    DataGroupDict,
    FrequencyCoordArray,
    ImageXds,
    SkyArray,
    check_image,
)
from xradio.schema.check import check_dataset, check_datatree, check_dict
from xradio.testing.image import create_empty_test_image, download_image, remove_path

pytestmark = pytest.mark.usefixtures("dask_client_module")


def make_valid_image_xds() -> xr.Dataset:
    """Build a small synthetic image dataset that conforms to the image schema,
    mirroring what ``open_image`` produces for a CASA sky image."""

    nchan, npol, nl, nm = 2, 3, 4, 4
    frequency_values = np.array([1.412e9, 1.413e9])
    rest_frequency = 1.413e9

    coords = {
        "time": xr.DataArray(
            np.array([54000.1]),
            dims="time",
            attrs={"type": "time", "units": "d", "scale": "utc", "format": "mjd"},
        ),
        "frequency": xr.DataArray(
            frequency_values,
            dims="frequency",
            attrs={
                "rest_frequency": {
                    "data": rest_frequency,
                    "dims": [],
                    "attrs": {"units": "Hz", "type": "quantity"},
                },
                "reference_frequency": {
                    "attrs": {
                        "units": "Hz",
                        "observer": "lsrk",
                        "type": "spectral_coord",
                    },
                    "data": float(frequency_values[0]),
                    "dims": [],
                },
                "type": "spectral_coord",
                "units": "Hz",
                "frame": "LSRK",
                "wave_units": "mm",
            },
        ),
        "velocity": xr.DataArray(
            (1 - frequency_values / rest_frequency) * 2.99792458e8,
            dims="frequency",
            attrs={"doppler_type": "radio", "units": "m/s", "type": "doppler"},
        ),
        "polarization": xr.DataArray(["I", "Q", "U"], dims="polarization"),
        "l": xr.DataArray(
            np.linspace(2e-5, -2e-5, nl), dims="l", attrs={"note": "AIPS Memo #27"}
        ),
        "m": xr.DataArray(
            np.linspace(-2e-5, 2e-5, nm), dims="m", attrs={"note": "AIPS Memo #27"}
        ),
        "beam_params_label": xr.DataArray(
            ["major", "minor", "pa"], dims="beam_params_label"
        ),
    }

    sky = xr.DataArray(
        np.zeros((1, nchan, npol, nl, nm), dtype=np.float32),
        dims=("time", "frequency", "polarization", "l", "m"),
        attrs={
            "type": "sky",
            "units": "Jy/beam",
            "telescope": {
                "name": "ALMA",
                "direction": {
                    "attrs": {
                        "coordinate_system": "geocentric",
                        "frame": "ITRF",
                        "origin_object_name": "earth",
                        "type": "location",
                        "units": "rad",
                    },
                    "data": [-1.18, -0.4],
                    "dims": ["ellipsoid_dir_label"],
                    "coords": {
                        "ellipsoid_dir_label": {
                            "dims": ["ellipsoid_dir_label"],
                            "data": ["lon", "lat"],
                        }
                    },
                },
            },
            "obsdate": {
                "attrs": {
                    "units": "d",
                    "scale": "utc",
                    "format": "mjd",
                    "type": "time",
                },
                "data": 54000.1,
                "dims": [],
            },
            "pointing_center": {
                "attrs": {"frame": "fk5", "type": "sky_coord", "units": "rad"},
                "data": [0.2, -0.5],
                "dims": "sky_dir_label",
                "coords": {
                    "sky_dir_label": {"data": ["ra", "dec"], "dims": "sky_dir_label"}
                },
            },
            "object_name": "test_object",
            "beam_fit_params": "BEAM_FIT_PARAMS_SKY",
        },
    )
    flag = xr.DataArray(
        np.zeros((1, nchan, npol, nl, nm), dtype=bool),
        dims=("time", "frequency", "polarization", "l", "m"),
        attrs={"type": "flag"},
    )
    beam_fit_params = xr.DataArray(
        np.zeros((1, nchan, npol, 3), dtype=np.float64),
        dims=("time", "frequency", "polarization", "beam_params_label"),
        attrs={"units": "rad", "type": "beam_fit_params_sky"},
    )

    attrs = {
        "coordinate_system_info": {
            "reference_direction": {
                "attrs": {
                    "frame": "fk5",
                    "type": "sky_coord",
                    "units": "rad",
                    "equinox": "j2000.0",
                },
                "data": [0.2, -0.5],
                "dims": "sky_dir_label",
                "coords": {
                    "sky_dir_label": {"data": ["ra", "dec"], "dims": "sky_dir_label"}
                },
            },
            "native_pole_direction": {
                "attrs": {
                    "frame": "NATIVE_PROJECTION",
                    "type": "location",
                    "units": "rad",
                },
                "data": [3.141592653589793, -0.5],
                "dims": "ellipsoid_dir_label",
                "coords": {
                    "ellipsoid_dir_label": {
                        "data": ["lon", "lat"],
                        "dims": "ellipsoid_dir_label",
                    }
                },
            },
            "projection": "SIN",
            "projection_parameters": [0.0, 0.0],
            "pixel_coordinate_transformation_matrix": [[1.0, 0.0], [0.0, 1.0]],
        },
        "type": "image_dataset",
        "data_groups": {
            "base": {
                "sky": "SKY",
                "flag": "FLAG_SKY",
                "beam_fit_params_sky": "BEAM_FIT_PARAMS_SKY",
            }
        },
    }

    return xr.Dataset(
        {"SKY": sky, "FLAG_SKY": flag, "BEAM_FIT_PARAMS_SKY": beam_fit_params},
        coords=coords,
        attrs=attrs,
    )


@pytest.fixture
def image_xds_valid():
    return make_valid_image_xds()


class TestImageSchemaSynthetic:
    """Schema checking of synthetic image datasets, including negative cases."""

    def test_valid_dataset_has_no_issues(self, image_xds_valid):
        assert not check_dataset(image_xds_valid, ImageXds)
        assert not check_image(image_xds_valid)

    def test_versioned_variables_are_checked(self, image_xds_valid):
        """A version of SKY (here SKY_MODEL) must be validated against the sky
        array schema via allow_multiple_versions."""
        xds = image_xds_valid
        xds["SKY_MODEL"] = xds["SKY"].astype(np.int32)
        xds["SKY_MODEL"].attrs = dict(xds["SKY"].attrs)
        issues = check_image(xds)
        assert issues, "int32 SKY_MODEL should have been flagged"
        assert any(i.path[-1] == ("dtype", None) for i in issues)

    def test_flag_variable_not_checked_as_sky(self, image_xds_valid):
        """FLAG_SKY contains 'SKY' as a substring but must not be validated
        against the sky array schema (it is bool, not float)."""
        assert not check_image(image_xds_valid)

    def test_flag_wrong_dtype(self, image_xds_valid):
        xds = image_xds_valid
        xds["FLAG_SKY"] = xds["FLAG_SKY"].astype(np.float32)
        xds["FLAG_SKY"].attrs = {"type": "flag"}
        issues = check_image(xds)
        assert any(i.path[-1] == ("dtype", None) for i in issues)

    def test_wrong_dataset_type(self, image_xds_valid):
        xds = image_xds_valid
        xds.attrs["type"] = "image"
        issues = check_image(xds)
        assert any(i.path[0] == ("attrs", "type") for i in issues)

    def test_missing_data_groups(self, image_xds_valid):
        xds = image_xds_valid
        del xds.attrs["data_groups"]
        issues = check_image(xds)
        assert any(i.path[0] == ("attrs", "data_groups") for i in issues)

    def test_data_group_references_missing_variable(self, image_xds_valid):
        xds = image_xds_valid
        xds.attrs["data_groups"]["base"]["sky"] = "SKY_NOT_THERE"
        issues = check_image(xds)
        assert any(
            i.path == [("data_vars", "SKY_NOT_THERE")] and "missing" in i.message
            for i in issues
        )

    def test_data_group_role_with_wrong_value_type(self, image_xds_valid):
        xds = image_xds_valid
        xds.attrs["data_groups"]["base"]["sky"] = 123
        issues = check_image(xds)
        # Exactly one issue: the 'base' group is validated once (through the
        # DataGroupsDict attribute schema), not double reported by check_image
        assert len(issues) == 1
        assert issues[0].path[-1] == ("", "sky")

    def test_unknown_data_group_role(self, image_xds_valid):
        """A misspelled role would silently escape validation of the variable
        it references, so check_image reports unknown roles."""
        xds = image_xds_valid
        xds.attrs["data_groups"]["base"]["skyy"] = "SKY_NOT_THERE"
        issues = check_image(xds)
        assert len(issues) == 1
        assert "Unknown data group role" in issues[0].message

    def test_data_group_not_a_dictionary(self, image_xds_valid):
        xds = image_xds_valid
        xds.attrs["data_groups"]["base"] = "not_a_dict"
        issues = check_image(xds)
        assert any("is not a dictionary" in i.message for i in issues)

    def test_extra_data_group_is_checked(self, image_xds_valid):
        """Data groups other than 'base' are checked against DataGroupDict."""
        xds = image_xds_valid
        xds["SKY_IMPROVED"] = xds["SKY"]
        xds.attrs["data_groups"]["improved"] = {
            "sky": "SKY_IMPROVED",
            "flag": "FLAG_SKY",
            "description": "improved image",
            "date": "2024-06-10T00:00:00.000",
        }
        assert not check_image(xds)
        xds.attrs["data_groups"]["improved"]["flag"] = 3.14
        issues = check_image(xds)
        assert len(issues) == 1

    def test_bad_reference_direction_frame(self, image_xds_valid):
        xds = image_xds_valid
        csys = deepcopy(xds.attrs["coordinate_system_info"])
        csys["reference_direction"]["attrs"]["frame"] = "not_a_frame"
        xds.attrs["coordinate_system_info"] = csys
        issues = check_image(xds)
        assert any("Disallowed literal value" in i.message for i in issues)

    def test_bad_reference_direction_equinox(self, image_xds_valid):
        """The equinox of the reference direction is optional but must be a string."""
        xds = image_xds_valid
        csys = deepcopy(xds.attrs["coordinate_system_info"])
        csys["reference_direction"]["attrs"]["equinox"] = 2000.0
        xds.attrs["coordinate_system_info"] = csys
        issues = check_image(xds)
        assert any(i.path[-1] == ("attrs", "equinox") for i in issues)

    def test_delete_data_variables_keeps_groups_consistent(self, image_xds_valid):
        """Deleting a variable through the accessor also removes the data
        group entries referencing it, so the dataset still checks clean."""
        xds = image_xds_valid.xr_img.delete_data_variables(["BEAM_FIT_PARAMS_SKY"])
        assert "beam_fit_params_sky" not in xds.attrs["data_groups"]["base"]
        assert not check_image(xds)

    def test_bad_projection_parameters(self, image_xds_valid):
        xds = image_xds_valid
        csys = deepcopy(xds.attrs["coordinate_system_info"])
        csys["projection_parameters"] = ["zero", "zero"]
        xds.attrs["coordinate_system_info"] = csys
        issues = check_image(xds)
        assert any("list of floats" in i.message for i in issues)

    def test_bad_transformation_matrix(self, image_xds_valid):
        xds = image_xds_valid
        csys = deepcopy(xds.attrs["coordinate_system_info"])
        csys["pixel_coordinate_transformation_matrix"] = [1.0, 0.0]
        xds.attrs["coordinate_system_info"] = csys
        issues = check_image(xds)
        assert any("list of lists of floats" in i.message for i in issues)

    def test_missing_frequency_measures(self, image_xds_valid):
        xds = image_xds_valid
        xds.coords["frequency"].attrs.pop("rest_frequency")
        issues = check_image(xds)
        assert any(i.path[-1] == ("attrs", "rest_frequency") for i in issues)

    def test_missing_frequency_units_and_frame(self, image_xds_valid):
        """The frequency units and frame attributes are required."""
        xds = image_xds_valid
        xds.coords["frequency"].attrs.pop("units")
        xds.coords["frequency"].attrs.pop("frame")
        issues = check_image(xds)
        paths = {i.path[-1] for i in issues}
        assert ("attrs", "units") in paths
        assert ("attrs", "frame") in paths

    def test_check_datatree_dispatch(self, image_xds_valid):
        """The image schema is registered under the 'image_dataset' type, so
        check_datatree validates image nodes without being told the schema."""
        dt = xr.DataTree(dataset=image_xds_valid)
        assert not check_datatree(dt)

        invalid = make_valid_image_xds()
        invalid["FLAG_SKY"] = invalid["FLAG_SKY"].astype(np.float32)
        invalid["FLAG_SKY"].attrs = {"type": "flag"}
        dt_invalid = xr.DataTree(dataset=invalid)
        assert check_datatree(dt_invalid)

    def test_data_group_dict_check(self):
        assert not check_dict({"sky": "SKY"}, DataGroupDict)
        issues = check_dict({"sky": 1}, DataGroupDict)
        assert issues


class TestImageSchemaConstructors:
    """The schema classes double as constructors that validate on creation."""

    def test_frequency_coord_constructor(self):
        freq = FrequencyCoordArray(
            [1.412e9, 1.413e9],
            rest_frequency={
                "data": 1.413e9,
                "dims": [],
                "attrs": {"units": "Hz", "type": "quantity"},
            },
            reference_frequency={
                "attrs": {"units": "Hz", "observer": "lsrk", "type": "spectral_coord"},
                "data": 1.412e9,
                "dims": [],
            },
            frame="LSRK",
        )
        assert isinstance(freq, xr.DataArray)
        assert freq.attrs["type"] == "spectral_coord"
        assert freq.attrs["units"] == "Hz"
        assert freq.attrs["frame"] == "LSRK"

    def test_sky_array_constructor(self, image_xds_valid):
        sky = SkyArray(
            np.zeros((1, 2, 3, 4, 4), dtype=np.float32),
            time=image_xds_valid.time,
            frequency=image_xds_valid.frequency,
            polarization=image_xds_valid.polarization,
            l=image_xds_valid.l,
            m=image_xds_valid.m,
            units="Jy/beam",
        )
        assert isinstance(sky, xr.DataArray)
        assert sky.attrs["type"] == "sky"


class TestMakeEmptyImageSchemas:
    """The empty image factories produce datasets that conform to the schema."""

    @pytest.mark.parametrize(
        "factory,do_sky_coords",
        [
            pytest.param(make_empty_sky_image, True, id="sky"),
            pytest.param(make_empty_sky_image, False, id="sky_no_coords"),
            pytest.param(make_empty_aperture_image, None, id="aperture"),
            pytest.param(make_empty_lmuv_image, True, id="lmuv"),
        ],
    )
    def test_make_empty_image_schema(self, factory, do_sky_coords):
        xds = create_empty_test_image(factory, do_sky_coords)
        issues = check_image(xds)
        assert not issues, f"Schema check of empty image failed: {issues}"
        assert not check_datatree(xr.DataTree(dataset=xds))


class TestImageSchemaFromFormats:
    """Schema checking of images opened from CASA, FITS and zarr stores."""

    _casa_image = "casa_test_image.im"
    _fits_image = "test_image.fits"
    _uv_image = "complex_valued_uv.im"
    _zarr_store = "test_image_schema_write.zarr"

    @classmethod
    def setup_class(cls):
        for fname in [cls._casa_image, cls._fits_image, cls._uv_image]:
            download_image(fname)

    @classmethod
    def teardown_class(cls):
        for path in [cls._casa_image, cls._fits_image, cls._uv_image, cls._zarr_store]:
            remove_path(path)

    def test_open_casa_image_schema(self):
        xds = open_image(self._casa_image)
        issues = check_image(xds)
        assert not issues, f"Schema check of CASA image failed: {issues}"

    def test_open_fits_image_schema(self):
        xds = open_image(self._fits_image)
        issues = check_image(xds)
        assert not issues, f"Schema check of FITS image failed: {issues}"

    def test_open_aperture_image_schema(self):
        xds = open_image(self._uv_image)
        issues = check_image(xds)
        assert not issues, f"Schema check of aperture image failed: {issues}"

    def test_multi_image_open_schema(self):
        """Opening multiple images (with versioned sky variables) yields a
        dataset that conforms to the schema."""
        xds = open_image(
            {
                "sky_deconvolved": self._casa_image,
                "sky_dirty": self._casa_image,
            }
        )
        assert "SKY_DECONVOLVED" in xds.data_vars
        assert "SKY_DIRTY" in xds.data_vars
        issues = check_image(xds)
        assert not issues, f"Schema check of multi image dataset failed: {issues}"

    def test_zarr_roundtrip_schema(self):
        xds = open_image(self._casa_image)
        remove_path(self._zarr_store)
        write_image(xds, self._zarr_store, out_format="zarr", overwrite=True)
        zarr_xds = open_image(self._zarr_store)
        issues = check_image(zarr_xds)
        assert not issues, f"Schema check of zarr image failed: {issues}"
