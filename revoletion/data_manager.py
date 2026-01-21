"""
Module providing an interface to retrieve and manage PV data.
"""

import abc
import enum
import json
import logging
import pathlib
import typing

import pandas as pd
import pvlib
import pydantic
import requests
import typing_extensions

from . import location, time, utils

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
    def __init__(self, location: location.Location, api_name: str, msg: str) -> None:
        super().__init__(f"Failed to fetch timeseries data for location {location} from {api_name} API: {msg}")


class PvArray(pydantic.BaseModel):
    """
    Holds relevant specification about the installation of a PV array.

    Required by the different API implementations to determine the amount of generated power.
    """

    tracking_type: int
    horizon: bool
    mounting_place: str
    pv_tech: str
    rad_database: str
    tilt: typing.Literal["optimal"] | float | None = None
    azimuth: typing.Literal["optimal"] | float | None = None
    horizon_custom: list[float] | None = None


class DataProvider(abc.ABC):
    """Base class for a provider which can load or retrieve.

    Each provider has methods to load (`load_data_from_file`) and request (`request_data_from_api`) the raw provider data as well as remapping logic (`remap_data`).
    The remapping is done to transform the raw provider data to a consistent representation for REVOL-E-TION.

    A consumer is expected to explicitly request the remapping of the retrieved data. This enables caching of the raw data, e.g., for large scale mode.
    """

    def __init__(self, logger: logging.Logger, **_) -> None:
        self._logger = logger

    @abc.abstractmethod
    def request_data_from_api(
        self, location: location.Location, start: pd.Timestamp, end: pd.Timestamp, array: PvArray
    ) -> pd.DataFrame: ...

    @abc.abstractmethod
    def load_data_from_file(
        self, file: pathlib.Path, location: location.Location, array: PvArray | None = None
    ) -> pd.DataFrame:
        """Read and load the timeseries data from a file."""
        ...

    @abc.abstractmethod
    def remap_data(
        self, data: pd.DataFrame, location: location.Location, timeframe: time.TimeFrame, **_
    ) -> pd.DataFrame:
        data = data.resample(timeframe.dti.freq).mean().ffill().bfill()
        # convert to local time
        data.index = data.index.tz_convert(tz=location.timezone)
        # slice data
        data = data.loc[timeframe.dti_extd, ["power_spec", "speed_wind", "temp_air"]]
        return data


class SolcastDataProvider(DataProvider):
    """Provider to request and load Solcast data."""

    _API_NAME = "Solcast"

    def __init__(self, logger: logging.Logger, api_key: str | None = None) -> None:
        super().__init__(logger)
        self._api_key = api_key

    @typing_extensions.override
    def request_data_from_api(
        self, location: location.Location, timeframe: time.TimeFrame, array: PvArray
    ) -> pd.DataFrame:
        if timeframe.start - timeframe.end > pd.Timedelta(days=31):
            raise DataProviderApiError(
                location=location, api_name=self._API_NAME, msg="Solcast API only supports 31 days at a time"
            )

        if self._api_key is None:
            raise DataProviderApiError(location=location, api_name=self._API_NAME, msg="no Solcast API key specified")

        if array.tracking_type not in _SOLCAST_TRACKING_TYPE_MAPPING:
            raise DataProviderApiError(
                location=location,
                api_name=self._API_NAME,
                msg=f"tracking type {array.tracking_type} cannot be mapped to valid Solcast array type",
            )
        array_type = _SOLCAST_TRACKING_TYPE_MAPPING[array.tracking_type]

        if (location.latitude != _SOLCAST_UNMETERED_LATITUDE) and (location.longitude != _SOLCAST_UNMETERED_LONGITUDE):
            self._logger.warning("metered Solcast location selected")

        params = dict(
            latitude=location.latitude,
            longitude=location.longitude,
            start=timeframe.start.isoformat(),
            end=timeframe.end.isoformat(),
            period=_SOLCAST_DEFAULT_PERIOD,
            output_parameters=_SOLCAST_DEFAULT_OUTPUT_PARAMETERS,
            format="json",
            array_type=array_type,
            time_zone="utc",
            include_metadata=False,
            terrain_shading=True,
        )

        if array.tilt is not None:
            params["tilt"] = array.tilt
        if array.azimuth is not None:
            # convert to Solcast convention: north=0, east=-90, south=180, west=90
            params["azimuth"] = x - 360 if (x := (-1 * array.azimuth) % 360) > 180 else x

        response = requests.get(
            url=_SOLCAST_API_BASE_URL,
            headers={"Authorization": f"Bearer {self._solcast_api_key}"},
            params=params,
        )

        if response.status_code != 200:
            raise DataProviderApiError(
                location=location,
                api_name=self._API_NAME,
                msg=f"Solcast API returned status {response.status_code}: {response.json()['response_status']['message']}",
            )

        try:
            json_data = response.json()
        except json.JSONDecodeError as e:
            raise DataProviderApiError(
                location=location,
                api_name=self._API_NAME,
                msg="response is not valid JSON",
            ) from e

        if "estimated_actuals" not in json_data:
            raise DataProviderApiError(
                location=location, api_name=self._API_NAME, msg="missing key 'estimated_actuals' in Solcast response"
            )

        return pd.json_normalize(json_data["estimated_actuals"])

    @typing_extensions.override
    def load_data_from_file(
        self,
        file: pathlib.Path,
    ) -> pd.DataFrame:
        return pd.read_csv(file)

    @typing_extensions.override
    def remap_data(
        self, data: pd.DataFrame, location: location.Location, timeframe: time.TimeFrame, array: PvArray
    ) -> pd.DataFrame:
        data.index = pd.to_datetime(data["period_end"]) - pd.to_timedelta(data["period"])
        data.index.name = "period_start"
        data.drop(columns=["period", "period_end"], inplace=True)
        data.rename(columns={"air_temp": "temp_air", "wind_speed_10m": "speed_wind"}, inplace=True)
        data = data.tz_convert(location.timezone)

        data = self._calc_specific_power(data=data, location=location, array=array)

        return super().remap_data(data=data, location=location, timeframe=timeframe, array=array)

    def calc_specific_power(data: pd.DataFrame, location: location.Location, array: PvArray) -> pd.DataFrame:
        """
        Calculate potential PV array power from Solcast data considering actual tilt and azimuth.

        Parameters:
        data: solcast data imported using import_solcast()
        latitude: location latitude in decimal degrees north of equator. South is negative
        longitude: loaction longitude in decimal degrees east of prime meridian
        timezone: location timezone in string format, e.g. "Europe/Berlin"
        azimuth: panel azimuth in degrees east of north (i.e. north=0, east=90, south=180, west=270), default None is optimum
        tilt: panel tilt in degrees up from horizontal, default None is optimum

        Algorithm and parameters (cSi panels) as per
        - Huld T., Friesen G., Skoczek A., Kenny R.P., Sample T., Field M., Dunlop E.D. A power-rating model for
            crystalline silicon PV modules, Solar Energy Materials & Solar Cells, 2011 95, 3359-3369 (efficiency model)
        - Faiman, D. Assessing the outdoor operating temperature of photovoltaic modules. Prog. Photovolt. Res. Appl.2008,
            16, 307–315 (temperature model)
        """

        if array.azimuth is None:
            azimuth = 0 if location.latitude < 0 else 180
        else:
            azimuth = array.azimuth

        if array.tilt is None:
            tilt = abs(location.latitude)
        else:
            tilt = array.tilt

        solar_position = pvlib.location.Location(
            latitude=location.latitude,
            longitude=location.longitude,
        ).get_solarposition(times=data.index, method="nrel_numpy")

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
            dni=data["dni"],
            ghi=data["ghi"],
            dhi=data["dhi"],
            dni_extra=pvlib.irradiance.get_extra_radiation(data.index),
            model="haydavies",  # 'haydavies', 'reindl', 'klucher', or 'isotropic'
            albedo=data["albedo"],
        )

        gti_eff = irradiance["poa_direct"] * iam + irradiance["poa_diffuse"]

        temp_module = pvlib.temperature.faiman(
            poa_global=gti_eff,
            temp_air=data["temp_air"],
            wind_speed=data["speed_wind"],
            u0=26.9,  # W/(˚C.m2) - cSi Free standing as in PVGIS
            u1=6.2,  # W.s/(˚C.m3) - cSi Free standing as in PVGIS
        )

        data["power_spec"] = pvlib.pvarray.huld(
            effective_irradiance=gti_eff,
            temp_mod=temp_module,
            pdc0=1.0,  # for specific power
            cell_type="cSi",
        ).clip(lower=0)

        return data


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
    def request_data_from_api(
        self, location: location.Location, timeframe: time.TimeFrame, array: PvArray
    ) -> pd.DataFrame:
        shift = self.calc_api_request_shift(timeframe=timeframe)

        optimal_tilt = True if array.tilt is None else False
        optimal_angles = True if array.azimuth is None else False

        if optimal_angles and not optimal_tilt:
            raise DataProviderApiError(
                location,
                self._API_NAME,
                "optimal azimuth requires optimal tilt",
            )

        if array.pv_tech not in _PVGIS_API_PV_TECH_MAPPING:
            raise DataProviderApiError(location, self._API_NAME, f"unknown PV tech {array.pv_tech}")
        pv_tech_pvgis = _PVGIS_API_PV_TECH_MAPPING[array.pv_tech]

        data, *_ = pvlib.iotools.get_pvgis_hourly(
            latitude=location.latitude,
            longitude=location.longitude,
            start=timeframe.start.tz_convert("utc").year + shift,
            end=timeframe.end.tz_convert("utc").year + shift,
            raddatabase=array.rad_database.upper(),  # PVGIS is case sensitive
            components=True,
            surface_tilt=array.tilt if array.tilt is not None else 0,  # numeric
            surface_azimuth=array.azimuth if array.azimuth is not None else 0,  # numeric
            outputformat="json",
            usehorizon=True,
            userhorizon=array.horizon_custom,
            pvcalculation=True,
            peakpower=1,  # for specific power
            pvtechchoice=pv_tech_pvgis,
            mountingplace=array.mounting_place,
            loss=0,  # calculated in system
            trackingtype=array.tracking_type,
            optimal_surface_tilt=optimal_tilt,
            optimalangles=optimal_angles,
            url=_PVGIS_API_BASE_URL,
            map_variables=True,
            timeout=30,  # default
        )

        return data

    @typing_extensions.override
    def load_data_from_file(
        self,
        file: pathlib.Path,
        location: location.Location,
    ) -> pd.DataFrame:
        data, meta = pvlib.iotools.read_pvgis_hourly(file, map_variables=True)

        if (location.latitude != meta["inputs"]["latitude"]) or (location.longitude != meta["inputs"]["longitude"]):
            self._logger.warning("PV file location does not equal scenario location")

        return data

    @typing_extensions.override
    def remap_data(self, data: pd.DataFrame, location: location.Location, timeframe: time.TimeFrame, **_):
        shift = self.calc_api_request_shift(timeframe=timeframe)

        data.rename(columns={"wind_speed": "speed_wind"}, inplace=True)
        data["power_spec"] = data["P"] / 1e3  # convert 1kWp power to specific
        data.index = data.index.round("h")  # PVGIS does not give time slots as full hours
        data.index = data.index - pd.DateOffset(years=shift)

        return super().remap_data(data=data, location=location, timeframe=timeframe)


class BasicFileProvider(DataProvider):
    @typing_extensions.override
    def remap_data(self, data: pd.DataFrame, location: location.Location, timeframe=time.TimeFrame) -> pd.DataFrame:
        return super().remap_data(data=data, location=location, timeframe=timeframe)

    @typing_extensions.override
    def request_data_from_api(self, **_) -> None:
        raise NotImplementedError(f"Cannot request timeseries data with {type(self)}")

    @typing_extensions.override
    def load_data_from_file(self, file: pathlib.Path, location: location.Location, **_) -> pd.DataFrame:
        return utils.read_timeseries_csv(
            path_input_file=file,
            timezone=location.timezone,
            multiheader=False,
        )


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
    _location: location.Location
    _logger: logging.Logger

    def __init__(self, location: location.Location, logger: logging.Logger | None = None) -> None:
        self._location = location
        self._logger = logger

    def get_for_data_source(
        self,
        data_source: DataSource,
        array: PvArray,
        file_path: pathlib.Path | None,
        timeframe: time.TimeFrame,
        **provider_kwargs,
    ) -> pd.DataFrame:
        data_provider_type = data_source.get_data_provider()
        data_provider = data_provider_type(logger=self._logger, location=self._location, **provider_kwargs)

        if data_source.is_file_source():
            if file_path is None:
                raise ValueError(f"Argument 'file_path' must not be None for file data source {data_source}")
            data = data_provider.load_data_from_file(file=file_path, location=self._location, array=array)
        else:
            data = data_provider.request_data_from_api(location=self._location, timeframe=timeframe, array=array)

        remapped_data = data_provider.remap_data(data=data, location=self._location, timeframe=timeframe, array=array)

        return remapped_data
