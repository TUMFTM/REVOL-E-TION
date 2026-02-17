import logging
from dataclasses import dataclass, field
from typing import Literal, Self

import geopy
import pytz
import timezonefinder

_LOGGER = logging.getLogger(__name__)


def get_timezone_from_lat_lon(
    latitude: float,
    longitude: float,
    logger: logging.Logger | None = None,
    errors: Literal["ignore", "raise"] = "raise",
) -> pytz.BaseTzInfo | None:
    logger = logger or _LOGGER

    tzf = timezonefinder.TimezoneFinder()
    timezone_raw = tzf.certain_timezone_at(lat=latitude, lng=longitude)
    if timezone_raw is None:
        msg = f"Failed to determine timezone at {latitude}/{longitude}"
        if errors == "ignore":
            logger.warning(msg)
            return None
        elif errors == "raise":
            raise ValueError(msg)
    return pytz.timezone(timezone_raw)


def reverse_geocode_location(
    latitude: float, longitude: float, logger: logging.Logger | None = None
) -> None | geopy.Location:
    logger = logger or _LOGGER

    geolocator = geopy.geocoders.Nominatim(user_agent="location_finder")
    try:
        return geolocator.reverse(query=(latitude, longitude), language="en", exactly_one=True)
    except (geopy.exc.GeocoderUnavailable, geopy.exc.GeocoderServiceError):
        logger.warning(f"Reverse geocoding failed for {latitude}/{longitude}.")
        return None


def get_country_state_from_geolocation(
    geo_location: geopy.Location, logger: logging.Logger | None = None
) -> tuple[str, str] | None:
    logger = logger or _LOGGER

    address = geo_location.raw.get("address", {})

    if "ISO3166-2-lvl4" in address:
        country, state = address["ISO3166-2-lvl4"].split("-")
    elif "ISO3166-2-lvl3" in address:
        country, state = address["ISO3166-2-lvl3"].split("-")
    else:
        country = address.get("country_code", None)
        state = address.get("state", None)

    if country is None:
        logger.warning(f"Failed to extract country/state from geolocation for {geo_location.address}.")

    return country, state


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    timezone: pytz.BaseTzInfo = field(default_factory=lambda: pytz.timezone("Europe/Berlin"))
    country: str = "DE"
    state: str = "BY"

    @classmethod
    def create_from_lat_lon(
        cls,
        latitude: float,
        longitude: float,
        country: str = None,
        state: str = None,
        logger: logging.Logger | None = None,  # ToDo: can this method implicitly inherit a logger?
    ) -> Self:
        logger = logger or _LOGGER

        timezone = get_timezone_from_lat_lon(latitude, longitude, logger=logger, errors="ignore")
        if timezone is None:
            # Default timezone cannot be accessed via cls.timezone as it is a field with default_factory.
            timezone = cls.__dataclass_fields__["timezone"].default_factory()
            logger.warning(f"Using default timezone ({timezone.zone})")

        if country is None:
            geo_location = reverse_geocode_location(latitude, longitude)

            if geo_location:
                country, state = get_country_state_from_geolocation(geo_location)

        if isinstance(country, str):
            country = country.upper()
        if isinstance(state, str):
            state = state.upper()

        if not country:
            logger.warning(f"Using default country ({cls.country})")
            country = cls.country
            if not state:  # state is optional but set, if default country is used
                logger.warning(f"Using default state ({cls.state})")
                state = cls.state

        return cls(latitude=latitude, longitude=longitude, timezone=timezone, country=country, state=state)
