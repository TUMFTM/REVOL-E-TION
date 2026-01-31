import logging
from dataclasses import dataclass, field

import geopy
import pytz
import timezonefinder
from typing_extensions import Self


@dataclass
class Location:
    latitude: float
    longitude: float
    timezone: pytz.BaseTzInfo = field(default_factory=lambda: pytz.timezone("Europe/Berlin"))
    country: str = "DE"
    state: str = "BY"

    @classmethod
    def create_from_lat_lon(
        cls, latitude: float, longitude: float, logger: logging.Logger, geocode: bool = True
    ) -> Self:
        tzf = timezonefinder.TimezoneFinder()
        timezone_raw = tzf.certain_timezone_at(lat=latitude, lng=longitude)
        if timezone_raw is None:
            raise ValueError(f"Failed to determine timezone at {latitude}/{longitude}")

        timezone = pytz.timezone(timezone_raw)

        if geocode:
            location = cls._reverse_geocode_location(latitude, longitude)
        else:
            location = None

        if location is None:
            location = cls(
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
            )
            if geocode:
                # Warning is only necessary if geocoding was requested.
                logger.warning(
                    f"Connection to Geocoder failed. "
                    f"Using default country ({location.country}) and state ({location.state})."
                )

            return location

        address = location.raw.get("address", {})

        if "ISO3166-2-lvl4" in address:
            country, state = address["ISO3166-2-lvl4"].split("-")
        elif "ISO3166-2-lvl3" in address:
            country, state = address["ISO3166-2-lvl3"].split("-")
        else:
            # fallback: try country_code + state name
            country = address.get("country_code", "").upper()
            state = address.get("state", "")

        return cls(latitude=latitude, longitude=longitude, timezone=timezone, country=country, state=state)

    @staticmethod
    def _reverse_geocode_location(latitude: float, longitude: float) -> None | geopy.Location:
        geolocator = geopy.geocoders.Nominatim(user_agent="location_finder")
        try:
            return geolocator.reverse(query=(latitude, longitude), language="en", exactly_one=True)
        except geopy.exc.GeocoderUnavailable:
            return None
