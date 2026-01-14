#!/usr/bin/env python3

import pathlib
import pandas as pd
import pvlib
import requests


def calc_pvgis_shift(
    time_start: pd.Timestamp,
    time_end: pd.Timestamp,
    scenario=None,
):
    """
    Calculate the necessary shift (integer) in years for a request to comply with PVGIS limitations.

    :param time_start: Start time of the request (tz aware)
    :type time_start: pd.Timestamp
    :param time_end: End time of the request (tz aware)
    :type time_end: pd.Timestamp
    """

    scn_startyear = time_start.tz_convert("utc").year
    scn_endyear = time_end.tz_convert("utc").year

    PVGIS_MAX_YEAR = 2023
    PVGIS_MIN_YEAR = 2005

    shift = 0

    if (scn_endyear - scn_startyear) > (PVGIS_MAX_YEAR - PVGIS_MIN_YEAR):
        raise ValueError("PVGIS API request exceeds maximum length of available data")
    elif scn_endyear > PVGIS_MAX_YEAR:  # PVGIS-SARAH3 only has data up to 2023
        shift = PVGIS_MAX_YEAR - scn_endyear
        if scenario is not None:
            scenario.logger.warning(
                f"PVGIS API request exceeds available endtime - request shifted by "
                f"{shift} year{'s' if abs(shift != 1) else ''}"
            )
    elif scn_startyear < PVGIS_MIN_YEAR:  # PVGIS-SARAH3 only has data from 2005
        shift = scn_startyear - PVGIS_MAX_YEAR
        if scenario is not None:
            scenario.logger.warning(
                f"PVGIS API request exceeds available starttime - request shifted by "
                f"{shift} year{'s' if abs(shift != 1) else ''}"
            )

    return shift


def get_pvgis_from_api(
    latitude: float,
    longitude: float,
    time_start: pd.Timestamp,
    time_end: pd.Timestamp,
    trackingtype: int = 0,
    azimuth: float = None,
    tilt: float = None,
    raddatabase: str = "PVGIS-SARAH3",
    use_horizon: bool = True,
    horizon_custom: list = None,
    pvtechchoice: str = "crystSi",
    mountingplace: str = "free",
    save: pathlib.Path = None,
    scenario: "revoletion.simulation.Scenario" = None,
):
    shift = calc_pvgis_shift(
        time_start=time_start,
        time_end=time_end,
        scenario=scenario,
    )

    optimal_tilt = True if tilt is None else False
    optimal_angles = True if azimuth is None else False

    if optimal_angles and not optimal_tilt:
        raise ValueError("Optimal azimuth requires optimal tilt as well")

    data, _ = pvlib.iotools.get_pvgis_hourly(
        latitude=latitude,
        longitude=longitude,
        start=time_start.tz_convert("utc").year + shift,
        end=time_end.tz_convert("utc").year + shift,
        # PVGIS API is case-sensitive and REVOL-E-TION inputs are lowered -> revert
        raddatabase=raddatabase,
        components=True,  # output solar radiation components (beam, diffuse, and reflected)
        surface_tilt=tilt if tilt is not None else 0,  # has to be numeric
        surface_azimuth=azimuth if azimuth is not None else 0,  # has to be numeric
        outputformat="json",
        usehorizon=use_horizon,
        userhorizon=horizon_custom,
        pvcalculation=True,
        peakpower=1,  # for specific power
        pvtechchoice=pvtechchoice,
        mountingplace=mountingplace,
        loss=0,
        trackingtype=trackingtype,
        optimal_surface_tilt=optimal_tilt,
        optimalangles=optimal_angles,
        url="https://re.jrc.ec.europa.eu/api/v5_3/",
        map_variables=True,
        timeout=30,  # default value
    )

    if save:
        data.to_csv(save, index=False)

    return data


def calc_specific_power_from_pvgis(
    data: pd.DataFrame,
    time_start: pd.Timestamp,
    time_end: pd.Timestamp,
):
    shift = calc_pvgis_shift(
        time_start=time_start,
        time_end=time_end,
    )

    data.rename(columns={"wind_speed": "speed_wind"}, inplace=True)
    data["power_spec"] = data["P"] / 1e3  # convert 1kWp power to specific
    data.index = data.index.round("h")  # PVGIS does not give time slots as full hours
    data.index = data.index - pd.DateOffset(years=shift)

    return data


def get_solcast_from_api(
    api_key: str,
    latitude: float,
    longitude: float,
    time_start: pd.Timestamp,
    time_end: pd.Timestamp,
    use_horizon: bool = True,
    trackingtype: int = 0,
    azimuth: float = None,
    tilt: float = None,
    save: pathlib.Path = False,
    scenario: "revoletion.simulation.Scenario" = None,
):
    if time_end - time_start > pd.Timedelta(days=31):
        raise NotImplementedError("Solcast API only supports 31 days at a time")

    if (api_key is None) and (scenario is not None):
        raise ValueError(f"Scenario {scenario.name}: no Solcast API key specified")

    if (latitude != 41.8902) and (longitude != 12.4922) and (scenario is not None):
        scenario.logger.warning("metered Solcast location selected")

    params = dict(
        latitude=latitude,
        longitude=longitude,
        start=time_start.isoformat(),
        end=time_end.isoformat(),
        period="PT5M",
        output_parameters=[
            "air_temp",
            "albedo",
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
        include_metadata=False,
        terrain_shading=use_horizon,
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

    data = pd.json_normalize(response.json()["estimated_actuals"])

    if save:  # catch false and None
        data.to_csv(save, index=False)

    return data


def calc_specific_power_from_solcast(
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
