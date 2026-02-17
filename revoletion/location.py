import logging
from dataclasses import dataclass, field

import geopy
import pytz
import timezonefinder
from typing_extensions import Self


def reverse_geocode_location(latitude: float, longitude: float) -> None | geopy.Location:
    geolocator = geopy.geocoders.Nominatim(user_agent="location_finder")
    try:
        return geolocator.reverse(query=(latitude, longitude), language="en", exactly_one=True)
    except geopy.exc.GeocoderUnavailable:
        return None
    except geopy.exc.GeocoderServiceError:
        return None


def get_country_state_from_geolocation(geo_location: geopy.Location) -> tuple[str, str] | None:
    address = geo_location.raw.get("address", {})

    if "ISO3166-2-lvl4" in address:
        country, state = address["ISO3166-2-lvl4"].split("-")
    elif "ISO3166-2-lvl3" in address:
        country, state = address["ISO3166-2-lvl3"].split("-")
    else:
        country = address.get("country_code", None)
        state = address.get("state", None)

        country = country.upper() if country else None

    return country, state


@dataclass
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
        logger: logging.Logger,
        country: str = None,
        state: str = None,
    ) -> Self:
        tzf = timezonefinder.TimezoneFinder()
        timezone_raw = tzf.certain_timezone_at(lat=latitude, lng=longitude)
        if timezone_raw is None:
            raise ValueError(f"Failed to determine timezone at {latitude}/{longitude}")

        timezone = pytz.timezone(timezone_raw)

        if country is None:
            geo_location = reverse_geocode_location(latitude, longitude)

            if geo_location:
                country, state = get_country_state_from_geolocation(geo_location)

                if country is None:
                    logger.warning(f"Failed to extract country/state from geolocation for {latitude}/{longitude}.")
            else:
                logger.warning(f"Reverse geocoding failed for {latitude}/{longitude}.")

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
