"""
Module providing an interface to retrieve and manage PV data.
"""

import abc
import enum
import json
import logging
import pathlib

import pandas as pd
import pvlib
import pydantic
import requests
import typing_extensions

from . import location as loc
from . import time, utils

_SOLCAST_API_BASE_URL = "https://api.solcast.com.au/data/historic/radiation_and_weather"
_SOLCAST_UNMETERED_LATITUDE = 41.8902
_SOLCAST_UNMETERED_LONGITUDE = 12.4922
_SOLCAST_DEFAULT_PERIOD = "PT5M"
_SOLCAST_DEFAULT_OUTPUT_PARAMETERS = [
    "air_temp",
    "albedo",
    "azimuth",
    "clearsky_dhi",
    "clearsky_dni",
    "clearsky_ghi",
    "clearsky_gti",
    "cloud_opacity",
    "dewpoint_temp",
    "dhi",
    "dni",
    "ghi",
    "gti",
    "precipitable_water",
    "precipitation_rate",
    "relative_humidity",
    "surface_pressure",
    "wind_direction_100m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_speed_10m",
    "zenith",
]
_SOLCAST_TRACKING_TYPE_MAPPING = {0: "fixed", 1: "horizontal_single_axis"}

_PVGIS_API_BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_3/"
_PVGIS_API_MAX_YEAR = 2023
_PVGIS_API_MIN_YEAR = 2005
_PVGIS_API_PV_TECH_MAPPING = {
    "crystsi": "crystSi",
    "cis": "CIS",
    "cdte": "CdTe",
    "unknown": "Unknown",
}


class DataProviderError(utils.RevoletionError): ...


class DataProviderApiError(DataProviderError):
    def __init__(self, location: loc.Location, api_name: str, msg: str) -> None:
        super().__init__(f"Failed to fetch timeseries data for location {location} from {api_name} API: {msg}")


class PvArray(pydantic.BaseModel):
    """
    Holds relevant specification about the installation of a PV array.

    Required by the different API implementations to determine the amount of generated power.
    """

    tracking_type: int = 0
    mounting_place: str = "free"
    type_cell: str | None = None
    rad_database: str | None = None
    tilt: float | None = None
    azimuth: float | None = None
    horizon_custom: list[float] | None = None


class DataProvider(abc.ABC):
    """Base class for a provider which can load or retrieve.

    Each provider has methods to load (`load_data_from_file`) and request (`request_data_from_api`) the raw provider data as well as remapping logic (`remap_data`).
    The remapping is done to transform the raw provider data to a consistent representation for REVOL-E-TION.

    A consumer is expected to explicitly request the remapping of the retrieved data. This enables caching of the raw data, e.g., for large scale mode.
    """

    def __init__(self, logger: logging.Logger, location: loc.Location, array: PvArray, **_):
        self.location = location
        self.array = array
        self.data = None

        self._logger = logger

    @abc.abstractmethod
    def request_data_from_api(self, timeframe: time.TimeFrame): ...

    @abc.abstractmethod
    def load_data_from_file(self, file: pathlib.Path): ...

    @abc.abstractmethod
    def remap_data(self, timeframe: time.TimeFrame) -> pd.DataFrame:
        self.data = self.data[["power_spec", "speed_wind", "temp_air"]]  # numeric data only for resampling
        self.data.index = self.data.index.tz_convert(tz=self.location.timezone)  # convert to local time
        self.data = self.data.resample(timeframe.dti.freq).mean().ffill().bfill()
        self.data = self.data.reindex(timeframe.dti_extd).interpolate(method="time")
        return self.data

    def calc_specific_power(self) -> pd.DataFrame:
        """
        Calculate potential PV array power from raw irradiation data considering actual tilt and azimuth.

        Algorithm and parameters (cSi panels) as per
        - Huld T., Friesen G., Skoczek A., Kenny R.P., Sample T., Field M., Dunlop E.D. A power-rating model for
            crystalline silicon PV modules, Solar Energy Materials & Solar Cells, 2011 95, 3359-3369 (efficiency model)
        - Faiman, D. Assessing the outdoor operating temperature of photovoltaic modules. Prog. Photovolt. Res. Appl.2008,
            16, 307–315 (temperature model)
        """

        if self.array.azimuth is None:
            azimuth = 0 if self.location.latitude < 0 else 180
        else:
            azimuth = self.array.azimuth

        if self.array.tilt is None:
            tilt = abs(self.location.latitude)
        else:
            tilt = self.array.tilt

        solar_position = pvlib.location.Location(
            latitude=self.location.latitude,
            longitude=self.location.longitude,
        ).get_solarposition(times=self.data.index, method="nrel_numpy")

        # angle of incidence
        angle_of_incidence = pvlib.irradiance.aoi(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solar_position["zenith"],
            solar_azimuth=solar_position["azimuth"],
        )

        # incidence angle modifier
        iam = pvlib.iam.martin_ruiz(angle_of_incidence, a_r=0.16)

        # global total irradiance
        irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solar_position["zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=self.data["dni"],
            ghi=self.data["ghi"],
            dhi=self.data["dhi"],
            dni_extra=pvlib.irradiance.get_extra_radiation(self.data.index),
            model="haydavies",  # 'haydavies', 'reindl', 'klucher', or 'isotropic'
            albedo=self.data["albedo"],
        )

        gti_eff = irradiance["poa_direct"] * iam + irradiance["poa_diffuse"]

        temp_module = pvlib.temperature.faiman(
            poa_global=gti_eff,
            temp_air=self.data["temp_air"],
            wind_speed=self.data["speed_wind"],
            u0=26.9,  # W/(˚C.m2) - cSi Free standing as in PVGIS
            u1=6.2,  # W.s/(˚C.m3) - cSi Free standing as in PVGIS
        )

        self.data["power_spec"] = pvlib.pvarray.huld(
            effective_irradiance=gti_eff,
            temp_mod=temp_module,
            pdc0=1.0,  # for specific power
            cell_type="cSi",  # todo enable other cell types
        ).clip(lower=0)


class SolcastDataProvider(DataProvider):
    """Provider to request and load Solcast data."""

    _API_NAME = "Solcast"

    def __init__(
        self, logger: logging.Logger, location: loc.Location, array: PvArray, api_key: str | None = None
    ) -> None:
        super().__init__(logger=logger, location=location, array=array)
        self._api_key = api_key

    @typing_extensions.override
    def request_data_from_api(self, timeframe: time.TimeFrame) -> pd.DataFrame:
        if timeframe.start - timeframe.end > pd.Timedelta(days=31):
            raise DataProviderApiError(
                location=self.location, api_name=self._API_NAME, msg="Solcast API only supports 31 days at a time"
            )

        if self._api_key is None:
            raise DataProviderApiError(
                location=self.location, api_name=self._API_NAME, msg="no Solcast API key specified"
            )

        if self.array.tracking_type not in _SOLCAST_TRACKING_TYPE_MAPPING:
            raise DataProviderApiError(
                location=self.location,
                api_name=self._API_NAME,
                msg=f"tracking type {self.array.tracking_type} cannot be mapped to valid Solcast array type",
            )
        tracking_type = _SOLCAST_TRACKING_TYPE_MAPPING[self.array.tracking_type]

        if (self.location.latitude != _SOLCAST_UNMETERED_LATITUDE) and (
            self.location.longitude != _SOLCAST_UNMETERED_LONGITUDE
        ):
            self._logger.warning("metered Solcast location selected")

        params = dict(
            latitude=self.location.latitude,
            longitude=self.location.longitude,
            start=timeframe.start.isoformat(),
            end=timeframe.end.isoformat(),
            period=_SOLCAST_DEFAULT_PERIOD,
            output_parameters=_SOLCAST_DEFAULT_OUTPUT_PARAMETERS,
            format="json",
            array_type=tracking_type,
            time_zone="utc",
            include_metadata=False,
            terrain_shading=True,
        )

        if self.array.tilt is not None:
            params["tilt"] = self.array.tilt
        if self.array.azimuth is not None:
            # convert to Solcast convention: north=0, east=-90, south=180, west=90
            params["azimuth"] = x - 360 if (x := (-1 * self.array.azimuth) % 360) > 180 else x

        response = requests.get(
            url=_SOLCAST_API_BASE_URL,
            headers={"Authorization": f"Bearer {self._solcast_api_key}"},
            params=params,
        )

        if response.status_code != 200:
            raise DataProviderApiError(
                location=self.location,
                api_name=self._API_NAME,
                msg=f"Solcast API returned status {response.status_code}: {response.json()['response_status']['message']}",
            )

        try:
            json_data = response.json()
        except json.JSONDecodeError as e:
            raise DataProviderApiError(
                location=self.location,
                api_name=self._API_NAME,
                msg="response is not valid JSON",
            ) from e

        if "estimated_actuals" not in json_data:
            raise DataProviderApiError(
                location=self.location,
                api_name=self._API_NAME,
                msg="missing key 'estimated_actuals' in Solcast response",
            )

        self.data = pd.json_normalize(json_data["estimated_actuals"])

    @typing_extensions.override
    def load_data_from_file(self, file: pathlib.Path):
        self.data = pd.read_csv(file)

    @typing_extensions.override
    def remap_data(self, timeframe: time.TimeFrame) -> pd.DataFrame:
        self.data.index = pd.to_datetime(self.data["period_end"]) - pd.to_timedelta(self.data["period"])
        self.data.index.name = "period_start"
        self.data.drop(columns=["period", "period_end"], inplace=True)
        self.data.rename(columns={"air_temp": "temp_air", "wind_speed_10m": "speed_wind"}, inplace=True)
        self.data = self.data.tz_convert(self.location.timezone)

        self.calc_specific_power()

        return super().remap_data(timeframe=timeframe)


class PvgisDataProvider(DataProvider):
    """Provider to request and load PVGIS data."""

    _API_NAME: str = "PVGIS"

    @typing_extensions.override
    def calc_api_request_shift(
        self,
        timeframe: time.TimeFrame,
        api_startyear: int = _PVGIS_API_MIN_YEAR,
        api_endyear: int = _PVGIS_API_MAX_YEAR,
    ) -> int:
        """
        Calculate the necessary shift (integer) in years for a request to comply with PVGIS limitations.
        """

        startyear = timeframe.start.tz_convert("utc").year
        endyear = timeframe.end.tz_convert("utc").year

        shift = 0

        if (endyear - startyear) > (api_endyear - api_startyear):
            raise ValueError("API request exceeds maximum length of available data")
        elif endyear > api_endyear:
            shift = api_endyear - endyear
            self._logger.warning(
                f"API request exceeds available endtime - request shifted by "
                f"{shift} year{'s' if abs(shift != 1) else ''}"
            )
        elif startyear < api_startyear:
            shift = startyear - api_startyear
            self._logger.warning(
                f"API request exceeds available starttime - request shifted by "
                f"{shift} year{'s' if abs(shift != 1) else ''}"
            )

        return shift

    @typing_extensions.override
    def request_data_from_api(self, timeframe: time.TimeFrame) -> pd.DataFrame:
        shift = self.calc_api_request_shift(timeframe=timeframe)

        optimal_tilt = True if self.array.tilt is None else False
        optimal_angles = True if self.array.azimuth is None else False

        if optimal_angles and not optimal_tilt:
            raise DataProviderApiError(
                self.location,
                self._API_NAME,
                "optimal azimuth requires optimal tilt",
            )

        if self.array.type_cell not in _PVGIS_API_PV_TECH_MAPPING:
            raise DataProviderApiError(self.location, self._API_NAME, f"unknown PV tech {self.array.type_cell}")
        pv_tech_pvgis = _PVGIS_API_PV_TECH_MAPPING[self.array.type_cell]

        self.data, *_ = pvlib.iotools.get_pvgis_hourly(
            latitude=self.location.latitude,
            longitude=self.location.longitude,
            start=timeframe.start.tz_convert("utc").year + shift,
            end=timeframe.end.tz_convert("utc").year + shift,
            raddatabase=self.array.rad_database.upper(),  # PVGIS is case sensitive
            components=True,
            surface_tilt=self.array.tilt if self.array.tilt is not None else 0,  # numeric
            surface_azimuth=self.array.azimuth if self.array.azimuth is not None else 0,  # numeric
            outputformat="json",
            usehorizon=True,
            userhorizon=self.array.horizon_custom,
            pvcalculation=True,
            peakpower=1,  # for specific power
            pvtechchoice=pv_tech_pvgis,
            mountingplace=self.array.mounting_place,
            loss=0,  # calculated in system
            trackingtype=self.array.tracking_type,
            optimal_surface_tilt=optimal_tilt,
            optimalangles=optimal_angles,
            url=_PVGIS_API_BASE_URL,
            map_variables=True,
            timeout=30,  # default
        )

    @typing_extensions.override
    def load_data_from_file(self, file: pathlib.Path) -> pd.DataFrame:
        self.data, meta = pvlib.iotools.read_pvgis_hourly(file, map_variables=True)

        if (self.location.latitude != meta["inputs"]["latitude"]) or (
            self.location.longitude != meta["inputs"]["longitude"]
        ):
            self._logger.warning("PV file location does not equal scenario location")

    @typing_extensions.override
    def remap_data(self, timeframe: time.TimeFrame, **_):
        shift = self.calc_api_request_shift(timeframe=timeframe)

        self.data.rename(columns={"wind_speed": "speed_wind"}, inplace=True)
        self.data["power_spec"] = self.data["P"] / 1e3  # convert 1kWp power to specific
        self.data.index = self.data.index.round("h")  # PVGIS does not give time slots as full hours
        self.data.index = self.data.index - pd.DateOffset(years=shift)

        return super().remap_data(timeframe=timeframe)


class BasicFileProvider(DataProvider):
    @typing_extensions.override
    def request_data_from_api(self, **_) -> None:
        raise NotImplementedError(f"Cannot request timeseries data with {type(self)}")

    @typing_extensions.override
    def load_data_from_file(self, file: pathlib.Path, location: loc.Location, **_) -> pd.DataFrame:
        self.data = utils.read_timeseries_csv(
            path_input_file=file,
            timezone=location.timezone,
            multiheader=False,
        )

    @typing_extensions.override
    def remap_data(self, data: pd.DataFrame, location: loc.Location, timeframe=time.TimeFrame) -> pd.DataFrame:
        return super().remap_data(data=data, location=location, timeframe=timeframe)


class DataSource(enum.Enum):
    SOLCAST_API = "solcast api"
    PVGIS_API = "pvgis api"
    PVGIS_FILE = "pvgis file"
    SOLCAST_FILE = "solcast file"
    TIME_SERIES_FILE = "file"

    def is_file_source(self) -> bool:
        return self in {DataSource.PVGIS_FILE, DataSource.SOLCAST_FILE, DataSource.TIME_SERIES_FILE}

    def get_data_provider(self) -> type[DataProvider]:
        match self:
            case DataSource.SOLCAST_API | DataSource.SOLCAST_FILE:
                return SolcastDataProvider
            case DataSource.PVGIS_API | DataSource.PVGIS_FILE:
                return PvgisDataProvider
            case DataSource.TIME_SERIES_FILE:
                return BasicFileProvider


class DataManager:
    _location: loc.Location
    _logger: logging.Logger

    def __init__(self, location: loc.Location, array: PvArray, logger: logging.Logger | None = None) -> None:
        self.location = location
        self.array = array
        self.data = None

        self._provider = None
        self._logger = logger

    def get_data(
        self,
        data_source: DataSource,
        file_path: pathlib.Path | None,
        timeframe: time.TimeFrame,
        **provider_kwargs,
    ) -> pd.DataFrame:
        data_provider_type = data_source.get_data_provider()
        self._provider = data_provider_type(
            logger=self._logger, location=self.location, array=self.array, **provider_kwargs
        )

        if data_source.is_file_source():
            if file_path is None:
                raise ValueError(f"Argument 'file_path' must not be None for file data source {data_source}")
            self._provider.load_data_from_file(file=file_path)
        else:
            self._provider.request_data_from_api(timeframe=timeframe)

        return self._provider.remap_data(timeframe=timeframe)
