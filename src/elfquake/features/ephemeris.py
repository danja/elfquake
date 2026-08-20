"""Low-precision solar and lunar ephemeris.

Everything here is a deterministic function of UTC, so these values are never
missing and never need a source connector. Accuracy is roughly two arcminutes
in lunar longitude and better than that for the Sun, which is far more than a
tidal-potential proxy or a phase-angle feature requires.

Positions are geocentric. Topocentric parallax shifts the lunar zenith angle by
up to about one degree, which changes the tidal term by well under a percent of
its range; it is not corrected for.

Reference: Meeus, *Astronomical Algorithms*, chapters 22, 25, 47 and 12, using
the truncated periodic series that those chapters give for reduced precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


J2000_JD = 2451545.0
JULIAN_CENTURY_DAYS = 36525.0

ASTRONOMICAL_UNIT_KM = 1.495978707e8
MOON_MEAN_DISTANCE_KM = 384400.0

# Solar tide relative to the lunar tide at their mean distances:
# (M_sun / M_moon) * (mean lunar distance / astronomical unit) ** 3.
SUN_MOON_MASS_RATIO = 2.7068510e7
SOLAR_TIDE_RELATIVE_COEFFICIENT = SUN_MOON_MASS_RATIO * (
    MOON_MEAN_DISTANCE_KM / ASTRONOMICAL_UNIT_KM
) ** 3

# Centre of the project's Italy bounding box, used as the reference site for
# the tidal potential. Italy spans about 13 degrees of longitude, so the
# semi-diurnal phase at the extremes differs from the centre by under an hour.
ITALY_REFERENCE_LATITUDE = 41.4
ITALY_REFERENCE_LONGITUDE = 12.5


@dataclass(frozen=True)
class BodyPosition:
    """Geocentric position of a body."""

    longitude_deg: float
    latitude_deg: float
    distance_km: float
    right_ascension_deg: float
    declination_deg: float


@dataclass(frozen=True)
class TidalPotential:
    """Degree-two tidal potential at a site, in units of the lunar mean term."""

    lunar: float
    solar: float

    @property
    def combined(self) -> float:
        return self.lunar + self.solar


def julian_day(moment: datetime) -> float:
    """Julian Day for a timezone-aware UTC datetime."""
    if moment.tzinfo is None:
        raise ValueError("julian_day requires a timezone-aware datetime")
    utc = moment.astimezone(timezone.utc)
    year = utc.year
    month = utc.month
    day = (
        utc.day
        + (utc.hour + (utc.minute + (utc.second + utc.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    )
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
    )


def julian_centuries(moment: datetime) -> float:
    return (julian_day(moment) - J2000_JD) / JULIAN_CENTURY_DAYS


def sun_position(moment: datetime) -> BodyPosition:
    t = julian_centuries(moment)
    mean_longitude = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
    mean_anomaly = 357.52911 + 35999.05029 * t - 0.0001537 * t * t
    eccentricity = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t
    anomaly = math.radians(mean_anomaly)
    centre = (
        (1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(anomaly)
        + (0.019993 - 0.000101 * t) * math.sin(2 * anomaly)
        + 0.000289 * math.sin(3 * anomaly)
    )
    true_longitude = mean_longitude + centre
    true_anomaly = math.radians(mean_anomaly + centre)
    radius_au = (1.000001018 * (1 - eccentricity**2)) / (
        1 + eccentricity * math.cos(true_anomaly)
    )
    return _equatorial(
        longitude_deg=_wrap_degrees(true_longitude),
        latitude_deg=0.0,
        distance_km=radius_au * ASTRONOMICAL_UNIT_KM,
        obliquity_deg=mean_obliquity_deg(t),
    )


def moon_position(moment: datetime) -> BodyPosition:
    t = julian_centuries(moment)
    # Meeus chapter 47 fundamental arguments.
    mean_longitude = 218.3164477 + 481267.88123421 * t - 0.0015786 * t * t
    elongation = 297.8501921 + 445267.1114034 * t - 0.0018819 * t * t
    sun_anomaly = 357.5291092 + 35999.0502909 * t - 0.0001536 * t * t
    moon_anomaly = 134.9633964 + 477198.8675055 * t + 0.0087414 * t * t
    argument_of_latitude = 93.2720950 + 483202.0175233 * t - 0.0036539 * t * t

    d = math.radians(elongation)
    m = math.radians(sun_anomaly)
    mp = math.radians(moon_anomaly)
    f = math.radians(argument_of_latitude)

    longitude = mean_longitude + (
        6.288774 * math.sin(mp)
        + 1.274027 * math.sin(2 * d - mp)
        + 0.658314 * math.sin(2 * d)
        + 0.213618 * math.sin(2 * mp)
        - 0.185116 * math.sin(m)
        - 0.114332 * math.sin(2 * f)
        + 0.058793 * math.sin(2 * d - 2 * mp)
        + 0.057066 * math.sin(2 * d - m - mp)
        + 0.053322 * math.sin(2 * d + mp)
        + 0.045758 * math.sin(2 * d - m)
        - 0.040923 * math.sin(m - mp)
        - 0.034720 * math.sin(d)
        - 0.030383 * math.sin(m + mp)
        + 0.015327 * math.sin(2 * d - 2 * f)
        - 0.012528 * math.sin(mp + 2 * f)
        + 0.010980 * math.sin(mp - 2 * f)
    )
    latitude = (
        5.128122 * math.sin(f)
        + 0.280602 * math.sin(mp + f)
        + 0.277693 * math.sin(mp - f)
        + 0.173237 * math.sin(2 * d - f)
        + 0.055413 * math.sin(2 * d - mp + f)
        + 0.046271 * math.sin(2 * d - mp - f)
        + 0.032573 * math.sin(2 * d + f)
        + 0.017198 * math.sin(2 * mp + f)
    )
    distance_km = (
        385000.56
        - 20905.355 * math.cos(mp)
        - 3699.111 * math.cos(2 * d - mp)
        - 2955.968 * math.cos(2 * d)
        - 569.925 * math.cos(2 * mp)
        + 246.158 * math.cos(2 * d - 2 * mp)
        - 204.586 * math.cos(2 * d - m)
        - 170.733 * math.cos(2 * d + mp)
        - 152.138 * math.cos(2 * d - m - mp)
        - 129.620 * math.cos(m - mp)
        + 108.743 * math.cos(d)
        + 104.755 * math.cos(m + mp)
    )
    return _equatorial(
        longitude_deg=_wrap_degrees(longitude),
        latitude_deg=latitude,
        distance_km=distance_km,
        obliquity_deg=mean_obliquity_deg(t),
    )


def mean_obliquity_deg(julian_centuries_value: float) -> float:
    t = julian_centuries_value
    return 23.439291 - 0.0130042 * t - 1.64e-7 * t * t + 5.04e-7 * t * t * t


def greenwich_mean_sidereal_time_deg(moment: datetime) -> float:
    jd = julian_day(moment)
    t = (jd - J2000_JD) / JULIAN_CENTURY_DAYS
    return _wrap_degrees(
        280.46061837
        + 360.98564736629 * (jd - J2000_JD)
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )


def moon_phase_angle_deg(moment: datetime) -> float:
    """Moon-Sun elongation in degrees: 0 at new moon, 180 at full moon.

    This increases monotonically through the synodic month, which is what makes
    it usable as a continuous feature. The named-phase category it replaces was
    a sawtooth over an arbitrary label ordering.
    """
    return _wrap_degrees(moon_position(moment).longitude_deg - sun_position(moment).longitude_deg)


def moon_illuminated_fraction(moment: datetime) -> float:
    moon = moon_position(moment)
    sun = sun_position(moment)
    elongation = math.acos(
        math.cos(math.radians(moon.latitude_deg))
        * math.cos(math.radians(moon.longitude_deg - sun.longitude_deg))
    )
    phase_angle = math.atan2(
        sun.distance_km * math.sin(elongation),
        moon.distance_km - sun.distance_km * math.cos(elongation),
    )
    return (1 + math.cos(phase_angle)) / 2


def tidal_potential(
    moment: datetime,
    *,
    latitude_deg: float = ITALY_REFERENCE_LATITUDE,
    longitude_deg: float = ITALY_REFERENCE_LONGITUDE,
) -> TidalPotential:
    """Degree-two tidal potential at a site.

    Units are the lunar term at the Moon's mean distance with the Moon
    overhead, so the lunar component runs about `-0.5` to `+1.1` and the solar
    component is roughly `0.46` times as large. Only ratios matter downstream,
    so no physical constants are carried.
    """
    moon = moon_position(moment)
    sun = sun_position(moment)
    sidereal = greenwich_mean_sidereal_time_deg(moment) + longitude_deg
    lunar = (MOON_MEAN_DISTANCE_KM / moon.distance_km) ** 3 * _legendre_p2(
        _cos_zenith_angle(moon, latitude_deg=latitude_deg, local_sidereal_deg=sidereal)
    )
    solar = (
        SOLAR_TIDE_RELATIVE_COEFFICIENT
        * (ASTRONOMICAL_UNIT_KM / sun.distance_km) ** 3
        * _legendre_p2(
            _cos_zenith_angle(sun, latitude_deg=latitude_deg, local_sidereal_deg=sidereal)
        )
    )
    return TidalPotential(lunar=lunar, solar=solar)


def _cos_zenith_angle(
    body: BodyPosition,
    *,
    latitude_deg: float,
    local_sidereal_deg: float,
) -> float:
    hour_angle = math.radians(local_sidereal_deg - body.right_ascension_deg)
    latitude = math.radians(latitude_deg)
    declination = math.radians(body.declination_deg)
    return math.sin(latitude) * math.sin(declination) + math.cos(latitude) * math.cos(
        declination
    ) * math.cos(hour_angle)


def _legendre_p2(cos_zenith: float) -> float:
    return (3 * cos_zenith * cos_zenith - 1) / 2


def _equatorial(
    *,
    longitude_deg: float,
    latitude_deg: float,
    distance_km: float,
    obliquity_deg: float,
) -> BodyPosition:
    longitude = math.radians(longitude_deg)
    latitude = math.radians(latitude_deg)
    obliquity = math.radians(obliquity_deg)
    right_ascension = math.atan2(
        math.sin(longitude) * math.cos(obliquity)
        - math.tan(latitude) * math.sin(obliquity),
        math.cos(longitude),
    )
    declination = math.asin(
        math.sin(latitude) * math.cos(obliquity)
        + math.cos(latitude) * math.sin(obliquity) * math.sin(longitude)
    )
    return BodyPosition(
        longitude_deg=longitude_deg,
        latitude_deg=latitude_deg,
        distance_km=distance_km,
        right_ascension_deg=_wrap_degrees(math.degrees(right_ascension)),
        declination_deg=math.degrees(declination),
    )


def _wrap_degrees(value: float) -> float:
    return value % 360.0
