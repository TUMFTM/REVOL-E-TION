#!/usr/bin/env python3

import pandas as pd
import pvlib
import requests


def get_pvgis_from_api(
    latitude: float,
    longitude: float,
    time_start: pd.Timestamp,
    time_end: pd.Timestamp,
    shading: bool,
    trackingtype: int,
    scenario=None,
    azimuth: float = None,
    tilt: float = None,
    raddatabase: str = "PVGIS-SARAH3",
    horizon=False,
    horizon_custom=False,
    pvtechchoice: str = "crystSi",
):
    scn_startyear = time_start.tz_convert("utc").year
    scn_endyear = time_end.tz_convert("utc").year

    PVGIS_MAX_YEAR = 2023
    PVGIS_MIN_YEAR = 2005

    req_shift_yrs = 0

    if (scn_endyear - scn_startyear) > (PVGIS_MAX_YEAR - PVGIS_MIN_YEAR):
        raise ValueError("PVGIS API request exceeds maximum length of available data")
    elif scn_endyear > PVGIS_MAX_YEAR:  # PVGIS-SARAH3 only has data up to 2023
        req_shift_yrs = PVGIS_MAX_YEAR - scn_endyear
        if scenario is not None:
            scenario.logger.warning(
                f"PVGIS API request exceeds available endtime - request shifted by "
                f"{req_shift_yrs} year{'s' if abs(req_shift_yrs != 1) else ''}"
            )
    elif scn_startyear < PVGIS_MIN_YEAR:  # PVGIS-SARAH3 only has data from 2005
        req_shift_yrs = scn_startyear - PVGIS_MAX_YEAR
        if scenario is not None:
            scenario.logger.warning(
                f"PVGIS API request exceeds available starttime - request shifted by "
                f"{req_shift_yrs} year{'s' if abs(req_shift_yrs != 1) else ''}"
            )

    req_startyear = scn_startyear + req_shift_yrs
    req_endyear = scn_endyear + req_shift_yrs

    optimal_tilt = True if tilt is None else False
    optimal_angles = True if azimuth is None else False
    if optimal_angles and not optimal_tilt:
        raise ValueError("Optimal azimuth requires optimal tilt as well")

    data, *_ = pvlib.iotools.get_pvgis_hourly(
        latitude=latitude,
        longitude=longitude,
        start=req_startyear,
        end=req_endyear,
        # PVGIS API is case-sensitive and REVOL-E-TION inputs are lowered -> revert
        raddatabase=raddatabase,
        components=True,  # output solar radiation components (beam, diffuse, and reflected)
        surface_tilt=tilt if tilt is not None else 0,  # has to be numeric
        surface_azimuth=azimuth if azimuth is not None else 0,  # has to be numeric
        outputformat="json",
        usehorizon=horizon,
        userhorizon=horizon_custom,
        pvcalculation=True,
        peakpower=1,
        # PVGIS API is case sensitive and all inputs are lowered -> revert
        pvtechchoice=pvtechchoice,
        mountingplace=self.mountingplace,
        loss=0,
        trackingtype=self.trackingtype,
        optimal_surface_tilt=optimal_tilt,
        optimalangles=optimal_angles,
        url="https://re.jrc.ec.europa.eu/api/v5_3/",
        map_variables=True,
        timeout=30,  # default value
    )

    self.data.rename(columns={"wind_speed": "speed_wind"}, inplace=True)

    self.data.index = self.data.index.round("h")  # PVGIS does not give time slots as full hours
    self.data.index = self.data.index - pd.DateOffset(years=req_shift_yrs)


def get_solcast_from_api(
    api_key: str,
    latitude: float,
    longitude: float,
    time_start: pd.Timestamp,
    time_end: pd.Timestamp,
    shading: bool,
    trackingtype: int,
    scenario=None,
    azimuth: float = None,
    tilt: float = None,
):
    if (api_key is None) and (scenario is not None):
        raise ValueError(f"Scenario {scenario.name}: no Solcast API key specified")

    if (latitude != 41.89021) and (longitude != 12.492231) and (scenario is not None):
        scenario.logger.warning("metered Solcast location selected")

    params = dict(
        latitude=latitude,
        longitude=longitude,
        start=time_start,
        end=time_end,
        period="PT5M",
        output_parameters=[
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
        ],
        format="json",
        array_type={0: "fixed", 1: "horizontal_single_axis"}[trackingtype],
        time_zone="utc",
        include_etadata=False,
        terrain_shading=shading,
    )

    if tilt is not None:
        params["tilt"] = tilt
    if azimuth is not None:
        # Convert to Solcast convention: (-180, 180], north=0, east=-90, south=180, west=90
        params["azimuth"] = x - 360 if (x := (-1 * azimuth) % 360) > 180 else x

    # get data from Solcast API
    response = requests.get(
        url="https://api.solcast.com.au/data/historic/radiation_and_weather",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
    )

    if response.status_code != 200:
        raise ValueError(
            f"Solcast API returned {response.status_code} instead of 200: "
            f"{response.json()['response_status']['message']}"
        )

    return pd.json_normalize(response.json()["estimated_actuals"])


def calc_specific_power_from_solcast(
    data: pd.DataFrame,
    latitude: float,
    longitude: float,
    timezone: str,
    azimuth: float,
    tilt: float,
) -> pd.DataFrame:
    """
    Calculate potential PV array power from Solcast data considering actual tilt and azimuth.

    Parameters:
    data: solcast data imported using import_solcast()
    latitude: location latitude in decimal degrees north of equator. South is negative
    longitude: loaction longitude in decimal degrees east of prime meridian
    timezone: location timezone in string format, e.g. "Europe/Berlin"
    azimuth: panel azimuth in degrees east of north (i.e. north=0, east=90, south=180, west=270)
    tilt: panel tilt in degrees up from horizontal

    Algorithm and parameters (cSi panels) as per
    - Huld T., Friesen G., Skoczek A., Kenny R.P., Sample T., Field M., Dunlop E.D. A power-rating model for
        crystalline silicon PV modules, Solar Energy Materials & Solar Cells, 2011 95, 3359-3369 (efficiency model)
    - Faiman, D. Assessing the outdoor operating temperature of photovoltaic modules. Prog. Photovolt. Res. Appl.2008,
        16, 307–315 (temperature model)
    """

    data.index = pd.to_datetime(data["period_end"]) - pd.to_timedelta(data["period"])
    data.index.name = "period_start"
    data.drop(columns=["period", "weather_type", "period_end"], inplace=True)  # string columns
    data.rename(columns={"air_temp": "temp_air", "wind_speed_10m": "speed_wind"}, inplace=True)
    data = data.tz_convert(timezone)

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

    power_spec = pvlib.pvarray.huld(
        effective_irradiance=gti_eff,
        temp_mod=temp_module,
        pdc0=1.0,  # for specific power
        cell_type="cSi",
    ).clip(lower=0)

    return power_spec
