"""
Dimension names shared by the xradio schemas.

The dimensions here are ``time``, ``frequency``, ``polarization``,
``sky_dir_label``, ``sky_dis_label``, ``ellipsoid_dir_label``,
``ellipsoid_dis_label``, ``cartesian_pos_label``, ``antenna_name``,
``baseline_id`` and ``receptor_label``: the measurement set v4 schema
(``xradio.measurement_set.schema``), the image schema
(``xradio.image.schema``) and the shared measures
(``xradio.schema.measures``) import the ones they use from here.
``xradio.schema.measures`` and ``xradio.measurement_set.schema``, where
earlier releases defined them, re-export them.
"""

from __future__ import annotations

from typing import Literal

Time = Literal["time"]
""" Observation time dimension """
Frequency = Literal["frequency"]
""" Frequency dimension """
Polarization = Literal["polarization"]
""" Polarization dimension """
SkyDirLabel = Literal["sky_dir_label"]
""" Coordinate labels of sky directions (typically shape 2 and 'ra', 'dec') """
SkyDisLabel = Literal["sky_dis_label"]
""" Coordinate labels of sky distance (typically shape 1 and 'dist') """
EllipsoidDirLabel = Literal["ellipsoid_dir_label"]
""" Coordinate labels of the angles of spherical earth location data
(typically shape 2 and 'lon', 'lat'). The ``coordinate_system`` attribute of
the location measure says whether the latitude is geodetic or geocentric. """
EllipsoidDisLabel = Literal["ellipsoid_dis_label"]
""" Coordinate label of the radial component of spherical earth location data
(typically shape 1): 'height' above the ellipsoid for geodetic locations, or
'dist' from the geocenter for geocentric locations (see the
``coordinate_system`` attribute of the location measure). """
CartesianPosLabel = Literal["cartesian_pos_label"]
""" Coordinate labels of geocentric earth location data (typically shape 3 and 'x', 'y', 'z')"""
AntennaName = Literal["antenna_name"]
""" Antenna name dimension """
BaselineId = Literal["baseline_id"]
""" Baseline ID dimension """
ReceptorLabel = Literal["receptor_label"]
""" Receptor label dimension """

# Represents "no dimension", i.e. used for coordinates and data variables with
# zero dimensions.
ZD = tuple[()]
