"""
Module which provides an interface to retrieve and manage PV data.
"""

import abc
import enum
import json
import logging
import pathlib
import typing

import numpy as np
import pandas as pd
import pvlib
import pydantic
import requests
import typing_extensions

from . import utils


class DataProviderError(utils.RevoletionError): ...


class DataProviderApiError(DataProviderError):
    def __init__(self, location: utils.Location, api_name: str, msg: str) -> None:
        super().__init__(f"Failed to fetch timeseries data for location {location} from {api_name} API: {msg}")


class PvInstallationInfo(pydantic.BaseModel):
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

    Each provider has methods to load (`load_ts_data_from_file`) and request (`request_ts_data_from_api`) the raw provider data as well as remapping logic (`remap_raw_ts_data`).
    The remapping is done to transform the raw provider data to a consistent representation for REVOL-E-TION.

    A consumer is expected to explicitly request the remapping of the retrieved data. This enables caching of the raw data, e.g., for large scale mode.
    """

    def __init__(self, logger: logging.Logger, **_) -> None:
        self._logger = logger

    @abc.abstractmethod
    def remap_raw_ts_data(
        self, raw_data: pd.DataFrame, location: utils.Location, dti: pd.DatetimeIndex
    ) -> pd.DataFrame:

        # resample to timestep, fill NaN values with previous ones (or next ones, if not available
        time_step = utils.Timestep.from_dti(dti)
        data = raw_data.resample(time_step.td).mean().ffill().bfill()
        # convert to local time
        data.index = data.index.tz_convert(tz=location.timezone)

        # only keep relevant columns and timestamps
        data = data.loc[dti, ["power_spec", "speed_wind", "temp_air"]]
        return data

    @abc.abstractmethod
    def request_ts_data_from_api(
        self, location: utils.Location, start: pd.Timestamp, end: pd.Timestamp, info: PvInstallationInfo
    ) -> pd.DataFrame: ...

    @abc.abstractmethod
    def load_ts_data_from_file(
        self, file: pathlib.Path, location: utils.Location, info: PvInstallationInfo | None = None
    ) -> pd.DataFrame:
        """Read and load the timeseries data from a file."""
        ...


# Solcast offers free results for the following coordinates.
_SOLCAST_UNMETERED_LATITUDE = 41.89021
_SOLCAST_UNMETERED_LONGITUDE = 12.492231
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
    "snow_depth",
    "snow_water_equivalent",
    "snow_soiling_rooftop",
    "snow_soiling_ground",
    "wind_direction_100m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_speed_10m",
    "zenith",
]
_SOLCAST_TRACKING_TYPE_MAPPING = {0: "fixed", 1: "horizontal_single_axis"}


class SolcastDataProvider(DataProvider):
    """Provider to request and load Solcast data."""

    _API_NAME = "Solcast"

    def __init__(self, logger: logging.Logger, solcast_api_key: str | None = None) -> None:
        super().__init__(logger)
        self._solcast_api_key = solcast_api_key

    @typing_extensions.override
    def remap_raw_ts_data(
        self, raw_data: pd.DataFrame, location: utils.Location, dti: pd.DatetimeIndex
    ) -> pd.DataFrame:
        solcast_data = raw_data.rename(
            columns={
                "air_temp": "temp_air",
                "wind_speed_10m": "speed_wind",
            },
        )
        period_timedelta = pd.to_timedelta(solcast_data["period"])
        period_end = pd.to_datetime(solcast_data["period_end"], utc=True)
        solcast_data["period_start"] = period_end - period_timedelta
        solcast_data.set_index(pd.DatetimeIndex(solcast_data["period_start"]), inplace=True)
        solcast_data = solcast_data.tz_convert(location.timezone)

        power = _calc_power_from_irradiation(
            solcast_data["gti"].values, solcast_data["temp_air"].values, solcast_data["speed_wind"].values
        )
        solcast_data["P"] = power

        return super().remap_raw_ts_data(solcast_data, location, dti)

    @typing_extensions.override
    def request_ts_data_from_api(
        self, location: utils.Location, start: pd.Timestamp, end: pd.Timestamp, info: PvInstallationInfo
    ) -> pd.DataFrame:
        # Cannot request timeseries data without an API key.
        if self._solcast_api_key is None:
            raise DataProviderApiError(location, self._API_NAME, "Missing Solcast API key")

        if location.latitude != _SOLCAST_UNMETERED_LATITUDE or location.longitude != _SOLCAST_UNMETERED_LONGITUDE:
            raise DataProviderApiError(
                location,
                self._API_NAME,
                "trying to request timeseries data for metered coordinates. Remove this line if you want to proceed with metered coordinates!",
            )

        if info.tracking_type not in _SOLCAST_TRACKING_TYPE_MAPPING:
            raise DataProviderApiError(
                location,
                self._API_NAME,
                "tracking type {info.tracking_type} cannot be mapped to valid solcast array type",
            )
        array_type = _SOLCAST_TRACKING_TYPE_MAPPING[info.tracking_type]

        params = dict(
            latitude=location.latitude,
            longitude=location.longitude,
            start=start,
            end=end,
            period=_SOLCAST_DEFAULT_PERIOD,
            output_parameters=_SOLCAST_DEFAULT_OUTPUT_PARAMETERS,
            format="json",
            array_type=array_type,
            time_zone="utc",
            include_etadata=False,
            terrain_shading=info.horizon,
        )

        if info.tilt != "optimal":
            params["tilt"] = info.tilt

        if info.azimuth != "optimal":
            params["azimuth"] = x - 360 if (x := (-1 * info.azimuth) % 360) > 180 else x

        response = requests.get(
            url="https://api.solcast.com.au/data/historic/radiation_and_weather",
            headers={"Authorization": f"Bearer {self._solcast_api_key}"},
            params=params,
        )
        if response.status_code != 200:
            raise DataProviderApiError(
                location,
                self._API_NAME,
                f"Solcast API returned status {response.status_code}: {response.json()['response_status']['message']}",
            )

        try:
            json_data = response.json()
        except json.JSONDecodeError as e:
            raise DataProviderApiError(
                location,
                self._API_NAME,
                "response is not valid JSON",
            ) from e

        if "estimated_actuals" not in json_data:
            raise DataProviderApiError(location, self._API_NAME, "missing key 'estimated_actuals' in Solcast response")

        data = pd.json_normalize(json_data["estimated_actuals"])
        return data

    @typing_extensions.override
    def load_ts_data_from_file(
        self, file: pathlib.Path, location: utils.Location, info: PvInstallationInfo | None = None
    ) -> pd.DataFrame:
        if info is None:
            raise DataProviderError(
                f"Cannot load data for location {location} from Solcast file: required PV installation info is missing"
            )

        data = pd.read_csv(file)

        # if at least one of azimuth or tilt are specified, recalculate irradiation for new pose
        if info.azimuth is None and info.tilt is None:
            return data

        if info.azimuth is None or info.azimuth == "optimal":
            azimuth = 0 if location.latitude < 0 else 180  # Solcast "optimum"
        else:
            azimuth = info.azimuth

        if info.tilt is None or info.tilt == "optimal":
            abs(location.latitude)  # Something close to Solcast "optimum"
        else:
            tilt = info.tilt

        # calculate solar position for location (gets altitude from lookup table)
        solar_position = pvlib.location.Location(
            latitude=location.latitude,
            longitude=location.longitude,
        ).get_solarposition(times=data.index, method="nrel_numpy")
        solar_azimuth = solar_position["azimuth"]
        solar_zenith = solar_position["zenith"]

        # alternatively use solcast data, but this data is rounded to integers  # ToDo: benchmark
        # solar_azimuth = self.data['azimuth']
        # solar_zenith = self.data['zenith']

        extra_radiation = pvlib.irradiance.get_extra_radiation(data.index)
        total_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
            dni=data["dni"],
            ghi=data["ghi"],
            dhi=data["dhi"],
            dni_extra=extra_radiation,
            model="haydavies",
            albedo=data["albedo"],
        )
        gti = total_irradiance["poa_global"]
        data["gti"] = gti

        return data[["temp_air", "speed_wind", "gti"]]


_PVGIS_API_BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_3/"
_PVGIS_API_MAX_YEAR = 2023
_PVGIS_API_MIN_YEAR = 2005
_PVGIS_API_MAX_LENGTH = _PVGIS_API_MAX_YEAR - _PVGIS_API_MIN_YEAR
_PVGIS_API_PV_TECH_MAPPING = {
    "crystsi": "crystSi",
    "cis": "CIS",
    "cdte": "CdTe",
    "unknown": "Unknown",
}


class PvgisDataProvider(DataProvider):
    """Provider to request and load PVGIS data."""

    _API_NAME: str = "PVGIS"
    _API_STARTYEAR: int = 2005
    _API_ENDYEAR: int = 2023

    @typing_extensions.override
    def calc_api_request_shift(
        self,
        time_start: pd.Timestamp,
        time_end: pd.Timestamp,
        api_startyear: int = _API_STARTYEAR,
        api_endyear: int = _API_ENDYEAR,
    ) -> int:
        """
        Calculate the necessary shift (integer) in years for a request to comply with PVGIS limitations.

        :param time_start: Start time of the request (tz aware)
        :type time_start: pd.Timestamp
        :param time_end: End time of the request (tz aware)
        :type time_end: pd.Timestamp
        """

        startyear = time_start.tz_convert("utc").year
        endyear = time_end.tz_convert("utc").year

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
    def remap_raw_ts_data(
        self,
        data: pd.DataFrame,
        time_start: pd.Timestamp,
        time_end: pd.Timestamp,
    ):
        shift = self.calc_api_request_shift(
            time_start=time_start,
            time_end=time_end,
        )

        data.rename(columns={"wind_speed": "speed_wind"}, inplace=True)
        data["power_spec"] = data["P"] / 1e3  # convert 1kWp power to specific
        data.index = data.index.round("h")  # PVGIS does not give time slots as full hours
        data.index = data.index - pd.DateOffset(years=shift)

        return super().remap_raw_ts_data(data, location, dti)

    @typing_extensions.override
    def request_ts_data_from_api(
        self, location: utils.Location, start: pd.Timestamp, end: pd.Timestamp, info: PvInstallationInfo
    ) -> pd.DataFrame:
        api_startyear = start.tz_convert("utc").year
        api_endyear = end.tz_convert("utc").year
        api_length = api_endyear - api_startyear
        api_shift = pd.to_timedelta("0 days")

        if api_length > _PVGIS_API_MAX_LENGTH:
            raise ValueError("PVGIS API request exceeds maximum length of available data")
        elif api_endyear > _PVGIS_API_MAX_YEAR:  # PVGIS-SARAH3 only has data up to 2023
            api_shift = pd.to_datetime(f"{_PVGIS_API_MAX_YEAR}-01-01 00:00:00+00:00") - pd.to_datetime(
                f"{api_endyear}-01-01 00:00:00+00:00"
            )
            api_endyear = _PVGIS_API_MAX_YEAR
            api_startyear = _PVGIS_API_MAX_YEAR - api_length
            self._logger.warning(
                f"PVGIS API request exceeds available endtime - data shifted by "
                f"{abs(api_shift)} year{'s' if abs(api_shift) == 1 else ''} to "
                f"end in {_PVGIS_API_MAX_YEAR}"
            )
        elif api_startyear < _PVGIS_API_MIN_YEAR:  # PVGIS-SARAH3 only has data from 2005
            api_shift = pd.to_datetime(f"{_PVGIS_API_MIN_YEAR}-01-01 00:00:00+00:00") - pd.to_datetime(
                f"{api_startyear}-01-01 00:00:00+00:00"
            )
            api_startyear = _PVGIS_API_MIN_YEAR
            api_endyear = _PVGIS_API_MIN_YEAR + api_length

            self._logger.warning(
                f"PVGIS API request exceeds available starttime - data shifted by "
                f"{abs(api_shift)} year{'s' if abs(api_shift) == 1 else ''} to "
                f"start in {_PVGIS_API_MIN_YEAR}"
            )

        optimal_tilt = True if info.tilt == "optimal" else False
        optimal_angles = True if info.azimuth == "optimal" else False
        if optimal_angles and not optimal_tilt:
            raise DataProviderApiError(
                location,
                self._API_NAME,
                f"optimal azimuth requires optimal tilt as well (azimuth={info.azimuth}; tilt={info.tilt})",
            )

        if info.pv_tech not in _PVGIS_API_PV_TECH_MAPPING:
            raise DataProviderApiError(location, self._API_NAME, f"unknown PV tech {info.pv_tech}")
        pv_tech_choice = _PVGIS_API_PV_TECH_MAPPING[info.pv_tech]

        pvgis_data, *_ = pvlib.iotools.get_pvgis_hourly(
            latitude=location.latitude,
            longitude=location.longitude,
            start=api_startyear,
            end=api_endyear,
            # PVGIS API is case sensitive and all inputs are lowered -> revert
            raddatabase=info.rad_database.upper(),
            components=True,  # output solar radiation components (beam, diffuse, and reflected)
            surface_tilt=info.tilt if info.tilt != "optimal" else 0,  # has to be numeric
            surface_azimuth=info.azimuth if info.azimuth != "optimal" else 0,  # has to be numeric
            outputformat="json",
            usehorizon=info.horizon,
            userhorizon=info.horizon_custom,
            pvcalculation=True,
            peakpower=1,
            # PVGIS API is case sensitive and all inputs are lowered -> revert
            pvtechchoice=pv_tech_choice,
            mountingplace=info.mounting_place,
            loss=0,
            trackingtype=info.tracking_type,
            optimal_surface_tilt=optimal_tilt,
            optimalangles=optimal_angles,
            url=_PVGIS_API_BASE_URL,
            map_variables=True,
            timeout=30,  # default value
        )

        pvgis_data.index = pvgis_data.index - api_shift

        return pvgis_data

    @typing_extensions.override
    def load_ts_data_from_file(
        self, file: pathlib.Path, location: utils.Location, info: PvInstallationInfo | None = None
    ) -> pd.DataFrame:
        data, meta = pvlib.iotools.read_pvgis_hourly(file, map_variables=True)

        # The location reset was kept during refactoring.
        # TODO: Why is it necessary to reset the location? Maybe replace it with validation instead.
        location.latitude = meta["inputs"]["latitude"]
        location.longitude = meta["inputs"]["longitude"]
        return data


class BasicFileProvider(DataProvider):
    @typing_extensions.override
    def remap_raw_ts_data(
        self, raw_data: pd.DataFrame, location: utils.Location, dti: pd.DatetimeIndex
    ) -> pd.DataFrame:
        return super().remap_raw_ts_data(raw_data, location, dti)

    @typing_extensions.override
    def request_ts_data_from_api(
        self, location: utils.Location, start: pd.Timestamp, end: pd.Timestamp, info: PvInstallationInfo
    ) -> pd.DataFrame:
        raise NotImplementedError(f"Cannot request timeseries data for {type(self)}")

    @typing_extensions.override
    def load_ts_data_from_file(
        self, file: pathlib.Path, location: utils.Location, info: PvInstallationInfo | None = None
    ) -> pd.DataFrame:
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
    _location: utils.Location
    _logger: logging.Logger

    def __init__(self, location: utils.Location, logger: logging.Logger | None = None) -> None:
        self._location = location
        self._logger = logger

    def get_for_data_source(
        self,
        data_source: DataSource,
        pv_installation_info: PvInstallationInfo,
        file_path: pathlib.Path | None,
        time_settings: utils.TimeSettings,
        **provider_kwargs,
    ) -> pd.DataFrame:
        data_provider_type = data_source.get_data_provider()
        data_provider = data_provider_type(self._logger, **provider_kwargs)

        if data_source.is_file_source():
            if file_path is None:
                raise ValueError(f"Argument `file_path` must not be None for file data source {data_source}")
            data = data_provider.load_ts_data_from_file(
                file=file_path, location=self._location, info=pv_installation_info
            )
        else:
            data = data_provider.request_ts_data_from_api(
                location=self._location, start=time_settings.start, end=time_settings.end, info=pv_installation_info
            )

        remapped_data = data_provider.remap_raw_ts_data(raw_data=data, location=self._location, dti=time_settings.dti)

        return remapped_data




def calc_specific_power_from_irradiation(
    data: pd.DataFrame,
    latitude: float,
    longitude: float,
    timezone: str,
    azimuth: float = None,
    tilt: float = None,
) -> pd.DataFrame:
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

    data.index = pd.to_datetime(data["period_end"]) - pd.to_timedelta(data["period"])
    data.index.name = "period_start"
    data.drop(columns=["period", "period_end"], inplace=True)  # string columns
    data.rename(columns={"air_temp": "temp_air", "wind_speed_10m": "speed_wind"}, inplace=True)
    data = data.tz_convert(timezone)

    if azimuth is None:
        azimuth = 0 if latitude < 0 else 180

    if tilt is None:
        tilt = abs(latitude)

    solar_position = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
    ).get_solarposition(times=data.index, method="nrel_numpy")

    # angle of incidence
    aoi = pvlib.irradiance.aoi(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position["zenith"],
        solar_azimuth=solar_position["azimuth"],
    )

    # incidence angle modifier
    iam = pvlib.iam.martin_ruiz(aoi, a_r=0.16)

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
