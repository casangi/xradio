from __future__ import annotations

from typing import Literal, Optional, Union, List
from xradio.schema.bases import (
    xarray_dataset_schema,
    xarray_dataarray_schema,
    dict_schema,
)
from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof, Name
import numpy


# Dimensions
Time = Literal["time"]
""" Observation time dimension """
Frequency = Literal["frequency"]
""" Frequency dimension """
FrequencySystemCal = Literal["frequency_system_cal"]
""" Frequency dimension in the system calibration dataset """
Polarization = Literal["polarization"]
""" Polarization dimension """
UvwLabel = Literal["uvw_label"]
""" Coordinate dimension of UVW data (typically shape 3 for 'u', 'v', 'w') """
SkyDirLabel = Literal["sky_dir_label"]
""" Coordinate labels of sky directions (typically shape 2 and 'ra', 'dec') """
LocalSkyDirLabel = Literal["local_sky_dir_label"]
""" Coordinate labels of local sky directions (typically shape 2 and 'az', 'alt') """
SphericalDirLabel = Literal["spherical_dir_label"]
""" Coordinate labels of spherical directions (shape 2 and 'lon', 'lat1' """
SkyPosLabel = Literal["sky_pos_label"]
""" Coordinate labels of sky positions (typically shape 3 and 'ra', 'dec', 'dist') """
SphericalPosLabel = Literal["spherical_pos_label"]
""" Coordinate labels of spherical positions (shape shape 3 and 'lon', 'lat1', 'dist2') """
EllipsoidPosLabel = Literal["ellipsoid_pos_label"]
""" Coordinate labels of geodetic earth location data (typically shape 3 and 'lon', 'lat', 'height')"""
CartesianPosLabel = Literal["cartesian_pos_label"]
""" Coordinate labels of geocentric earth location data (typically shape 3 and 'x', 'y', 'z')"""
nPolynomial = Literal["n_polynomial"]
""" For data that is represented as variable in time using Taylor expansion """
PolyTerm = Literal["poly_term"]
""" Polynomial term used in VLBI GAIN_CURVE """
LineLabel = Literal["line_label"]
""" Line labels (for line names and variables). """
FieldName = Literal["field_name"]
""" Field names dimension. """
