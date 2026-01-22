import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from revoletion import utils

_LOGGER = logging.getLogger(__name__)


class RevoletionBaseModel(BaseModel):
    _revoletion_docs_title: str
    """Specify the title that is displayed in the documentation section about this model."""

    _revoletion_docs_icon: str
    """Specify the icon that is displayed in the documentation section about this model."""

    @field_validator("*", mode="before")
    @classmethod
    def _infer_dtype(cls, value: Any) -> Any:
        """Infer the data type of a value from its string representation.

        Used to correctly convert the raw values from scenario specifications to python datatypes.
        E.g., `blocks` are specified as a dict espaced as a string in the scenario file.
        This validator ensures that those values are correctly converted to the corresponding pyhon dict and not stored as string.

        :param value: String value or value which can be converted to string.
        :returns: The corresponding parsed value.
        """
        return utils.infer_dtype(value)


class ScenarioModel(RevoletionBaseModel):
    """
    Each run can contain multiple scenario objects.
    A scenario object holds several parameters that are used by multiple blocks in the scenario.
    After a successful optimization it also contains the aggregated techno-economic results such as energy throughput, costs, revenues as well as LCOE, NPC, and NPV.
    """

    _revoletion_docs_title: str = "Scenario"
    _revoletion_docs_icon: str = "📋"

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    starttime: str = Field(
        title="Start Time",
        description="Start time of the project and the simulation in local time. If no time is given in addition to the date the project starts at 00:00 local time",
        json_schema_extra={"valid_values_or_format": "'dd.mm.YYYY' or 'dd.mm.YYYY HH:MM'"},
    )
    timestep: str = Field(
        title="Time step",
        description="Time step used for the simulation",
        json_schema_extra={
            "valid_values_or_format": "Formats compatible with pd.to_timedelta() such as 15min, 1h, 1D."
        },
    )
    sim_duration: int | str | None = Field(
        title="Project duration",
        description="Simulation duration. If given as integer the number is interpreted as number of days. Specifying a pandas.Timedelta() compliant string is also supported. The duration is rounded down to the specified timestep.",
        json_schema_extra={
            "valid_values_or_format": "[1, inf[ or strings such as '1 day 12 hours 14 minutes'",
            "not_required_for": "`sim_endtime` is given",
        },
    )
    sim_endtime: str | None = Field(
        title="Simulation end time",
        description="End time of the simulation in local time. If no time is given in addition to the date the simulation ends at 00:00 local time. Only one of the parameters sim_duration and sim_endtime can be specified. The other one has to be None.",
        json_schema_extra={
            "valid_values_or_format": "'dd.mm.YYYY' or 'dd.mm.YYYY HH:MM' or None",
            "not_required_for": "`sim_duration` is given",
        },
    )
    prj_duration: int = Field(
        title="Project duration",
        description="Project duration in years to which the economic results of the simulation duration are extrapolated",
        json_schema_extra={"valid_values_or_format": "[1, inf["},
    )
    compensate_sim_prj: bool = Field(
        title="Specific Capex/Opex compensation trigger",
        description="Trigger whether to optimize for sim duration (False) or project duration (True)",
    )

    # Optimization strategy parameters.
    strategy: Literal["go", "rh"] = Field(
        title="Strategy",
        description="Optimization strategy",
        json_schema_extra={"valid_values_or_format": "'go' or 'rh' (global optimum, rolling horizon)"},
    )
    len_ph: float | str | None = Field(
        None,
        title="Prediction horizon length",
        description="Length of the prediction horizon in hours. Will be rounded down to specified timestep of simulation. It can be given as float, which is interpreted as number of hours or a pd.Timedelta readable string. Neglected for every optimization strategy other than 'rh'.",
        json_schema_extra={"valid_values_or_format": "]0, inf[ or string such as 1 day"},
    )
    len_ch: int | str | None = Field(
        None,
        title="Control horizon length",
        description="Length of the control horizon in hours. Will be rounded down to specified timestep of simulation. It can be given as float, which is interpreted as number of hours or a pd.Timedelta readable string. Neglected for every optimization strategy other than 'rh'.",
        json_schema_extra={"valid_values_or_format": "]0, inf[ or string such as 12 hours"},
    )
    truncate_ph: bool = Field(
        title="Truncate Prediction Horizon",
        description="Toggles whether to truncate predictions horizons to simulation end time when using 'rh' optimization strategy. If activated all horizons are truncated to the simulation end time. Deactivation requires additional input data for all Prediction Horizons even beyond the end of the simulation specified by starttime and sim_duration.",
    )
    invest_max: float | None = Field(
        None,
        title="Maximum initial investment costs",
        description="Limit the initial investment costs to a specific amount. If no limit should be considered set this parameter to None.",
    )
    wacc: float | None = Field(
        None,
        title="Weighted average cost of capital",
        description="Weighted average cost of capital: discount rate for future expenses/revenues and energies per year.",
        ge=0,
        le=1,
        json_schema_extra={"valid_values_or_format": "[0, 1]"},
    )
    currency: str = Field(
        "EUR",
        title="Currency",
        description="Currency used to display results of economic calculations. No influence of calculation itself, only used for displaying.",
        json_schema_extra={"valid_values_or_format": "'str', e.g. 'EUR', 'USD'"},
    )
    latitude: float = Field(
        title="Latitude",
        description="Latitude of the location of the local energy system. Used to determine timezone, pv and wind data. Has to be given in WGS84",
        ge=-90,
        le=90,
    )
    longitude: float = Field(
        title="Longitude ",
        description="Longitude of the location of the local energy system. Used to determine timezone, pv and wind data. Has to be given in WGS84",
        ge=-90,
        le=90,
    )
    temp_air: float | str | None = Field(
        title="Air temperature",
        description="Air temperature. Can be given as string wih filename to csv file containing the columns 'time' (timezone aware timestamps) and 'temp_air' (temperature in °C), a float or int specifying a constant temperature in °C or the name of a PVSource.",
        json_schema_extra={
            "valid_values_or_format": "string with filename or name of PVSource instance or ]-inf, inf["
        },
    )
    cost_eps: float = Field(
        title="Epsilon costs",
        description="Cost added to some flows in order to disincentivice circular flows",
        ge=0,
    )
    blocks: dict[str, str] = Field(
        title="Blocks",
        description="All blocks present in the scenario except for the SystemCore, which is added automatically, in the format {block_name: class_name}. Non valid names are 'run', 'scenario' and 'core' (default name for block of class SystemCore).",
        json_schema_extra={"valid_values_or_format": "\"{'custom block name': 'class name of block'}\""},
    )

    @field_validator("starttime")
    @classmethod
    def validate_starttime(cls, v):
        """Validate that starttime is in the correct format."""
        try:
            # Try parsing with time
            _ = pd.to_datetime(v, format="%d.%m.%Y %H:%M")
        except ValueError:
            try:
                # Try parsing without time (will be set to 00:00)
                _ = pd.to_datetime(v, format="%d.%m.%Y")
            except ValueError:
                raise ValueError(f"Invalid 'starttime' {v} must be in format 'dd.mm.YYYY' or 'dd.mm.YYYY HH:MM'")
        return v

    @field_validator("timestep")
    @classmethod
    def validate_timestep(cls, v):
        """Validate that timestep can be converted to a pandas timedelta."""
        try:
            pd.to_timedelta(v)
        except ValueError:
            raise ValueError(f"timestep '{v}' cannot be converted to a pandas timedelta")
        return v

    @field_validator("sim_duration", "timestep")
    @classmethod
    def validate_pandas_timedelta(cls, value, info: ValidationInfo):
        """Validate simulation duration."""
        if isinstance(value, int) and value <= 0:
            raise ValueError(f"{info.field_name} as integer must be positive")
        elif isinstance(value, str):
            try:
                pd.to_timedelta(value)
            except ValueError:
                raise ValueError(f"{info.field_name} '{value}' cannot be converted to a pandas timedelta")
        return value


class SystemCoreModel(RevoletionBaseModel):
    """
    Collection of central energy system components (AC and DC buses and the two unidirectional transformers between them).
    This is present once and only once in every energy system defined in REVOL-E-TION under the name "core".
    For pure AC or DC systems, the respective core cost and size parameters can be set to zero to have no effect on the result.
    """

    _revoletion_docs_title: str = "SystemCore"
    _revoletion_docs_icon: str = "⇄"

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    size_preexisting_acdc: float | str = Field(
        title="Preexisting AC/DC size",
        description="Installed power of the AC/DC converter in the SystemCore in W. Set either size_preexisting_acdc or size_preexisting_dcac to 'equal' to set both preexisting converter sizes to the same value.",
        json_schema_extra={"valid_values_or_format": "[0, inf[ or 'equal'"},
    )
    capex_preexisting_acdc: bool = Field(
        title="Consideration of preexisting AC/DC size in capex",
        description="Trigger whether to consider preexisting component size specified in size_preexisting_acdc in initial capex calculation. Replacement capex are unaffected.",
    )
    size_max_acdc: float | None | Literal["equal"] = Field(
        default=None,
        title="Maximum size of AC/DC converter",
        description="Maximum size of the AC/DC converter of the SystemCore including preexisting size specified in size_preexisting_acdc. To enable unlimited investment set this parameter to None. Set either size_max_acdc or size_max_dcac to 'equal' to set both converters' maximum investments to the same value.",
        ge=0,
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None or 'equal'"},
    )
    invest_acdc: bool | Literal["equal"] = Field(
        title="Investment into AC/DC converter",
        description="Enable additional investment into the AC/DC converter of the SystemCore. Set either invest_acdc or invest_dcac to 'equal' to force the same expansion for both converters.",
    )
    size_preexisting_dcac: float | Literal["equal"] = Field(
        title="Existing DC/AC size",
        description="Installed power of the DC/AC converter in the SystemCore in W. Set either size_preexisting_acdc or size_preexisting_dcac to 'equal' to set both preexisting converter sizes to the same value.",
        json_schema_extra={"valid_values_or_format": "[0, inf[ or 'equal'"},
    )
    capex_preexisting_dcac: bool = Field(
        title="Consideration of preexisting DC/AC size in capex",
        description="Consider existing DC/AC size in initial capex calculation.",
    )
    size_max_dcac: float | None | Literal["equal"] = Field(
        default=None,
        title="Maximum size of DC/AC converter",
        description="Maximum size of the DC/AC converter of the SystemCore including preexisting size specified in `size_preexisting_dcac`. To enable unlimited investment set this parameter to None. Set either `size_max_acdc` or `size_max_dcac` to 'equal' to set both converters' maximum investments to the same value.",
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None or 'equal'"},
    )
    invest_dcac: bool | Literal["equal"] = Field(
        title="Investment into DC/AC converter",
        description="Enable additional investment into the DC/AC converter of the `SystemCore`. Set either `invest_acdc` or `invest_dcac` to 'equal' to force the same expansion for both converters.",
    )
    capex_spec: float = Field(
        title="Specific capital expenditures",
        description="Specific capital expenditures for each of the converters in the `SystemCore`: cost in currency per installed power (cumulative size of both converters) in W.",
        ge=0,
    )
    mntex_spec: float = Field(
        title="Specific maintenance expenditures",
        description="Specific maintenance expenditures for each of the converters in the `SystemCore`: cost in currency per year per installed power (cumulative size of both converters) in W.",
        ge=0,
    )
    opex_spec: float | str = Field(
        title="Specific operational expenditures",
        description="Specific operational expenditures for each of the converters in the `SystemCore` cost in currency per converted energy in Wh. Energy is measured at each converter's inflow. Can be given as float or filename of a csv file containing a timeseries.",
        json_schema_extra={"valid_values_or_format": "string with filename or [0, inf["},
    )
    ls: float = Field(
        title="Lifespan",
        description="Lifespan of the block in years after which it will be replaced.",
        ge=1,
    )
    ccr: float = Field(
        title="Cost change ratio",
        description="Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan.",
        ge=0,
    )
    eff_dcac: float = Field(
        title="DC/AC efficiency",
        description="Efficiency of the DC/AC converter in the `SystemCore`.",
        ge=0,
        le=1,
    )
    eff_acdc: float = Field(
        title="AC/DC efficiency",
        description="Efficiency of the AC/DC converter in the `SystemCore`.",
        ge=0,
        le=1,
    )


class FixedDemandModel(RevoletionBaseModel):
    """Undeferrable (i.e. inflexible) power demand such as households."""

    _revoletion_docs_title: str = "FixedDemand"
    _revoletion_docs_icon: str = "🏠"

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    load_profile: str = Field(
        title="Load Profile",
        description="Load profile for the fixed demand. Can be given as filename of a csv file containing a timeseries specifying the fixed demand of the block or as string defining a constant load or one of the standard load profiles by BDEW. If a filename is given, the file has to include the two columns 'time' and 'power' including a timezone aware timestamp and the corresponding power value in W",
        json_schema_extra={
            "valid_values_or_format": "string with filename, {'const', 'H0', 'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'L0', 'L1', 'L2'}"
        },
    )
    consumption_yrl: float = Field(
        title="Yearly consumption",
        description="Yearly consumption in Wh. Neglected if a filename is provided in load_profile.",
        ge=0,
    )
    system: Literal["ac", "dc"] = Field(
        description="The bus (AC or DC) the block is connected to.",
        title="System",
    )
    crev_spec: float | str = Field(
        title="Specific customer revenue",
        description="Specific customer revenue for consumed energy in currency per Wh. Can be given as float or filename of a csv file containing a timeseries.",
    )


class PVSourceModel(RevoletionBaseModel):
    """
    Photovoltaic array. Power potential is defined either using Solcast (pre-downloaded CSV file or API), PVGIS (pre-downloaded CSV file or API), or a timeseries CSV file.
    Although the Solcast API requires an active subscription plan, there is a limited free plan for researchers.
    """

    _revoletion_docs_title: str = "PVSource"
    _revoletion_docs_icon: str = "☀️"

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    size_preexisting_block: float = Field(
        title="Preexisting size",
        description="Installed peak power of the pv array in in W.",
        ge=0,
    )
    capex_preexisting_block: bool = Field(
        title="Consideration of preexisting block size in capex",
        description="Trigger whether to consider preexisting component size specified in `size_preexisting_block` in initial capex calculation. Replacement capex are unaffected.",
    )
    size_max_block: float | None = Field(
        title="Maximum size",
        description="Maximum size of `PVSource` including preexisting size specified in `size_preexisting_block`. To enable unlimited investment set this parameter to None.",
        ge=0,
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None"},
    )
    invest_block: bool = Field(
        title="Investment",
        description="Enable additional investment into the PV system.",
    )
    data_source: Literal["pvgis api", "solcast api", "pvgis file", "solcast file", "file"] = Field(
        title="Data source",
        description="Data source for pv power. This can be an API (PVGIS or Solcast) or a file containing data (PVGIS, Solcast or custom file). If Solcast API is chosen a valid API key has to specified in the run's arguments. A custom file has to include the columns 'time' (timezone aware timestamps), 'power_spec' (specific power in W per Wp), 'speed_wind' (in m/s), 'temp_air' (air temperature in °C).",
    )
    filename: str | None = Field(
        title="Filename",
        description="Name of a PVGIS, Solcast, or custom csv file if data_source is set to 'pvgis file', 'solcast file', or 'file', respectively. Otherwise set to None.",
        json_schema_extra={"valid_values_or_format": "filename or None"},
    )
    system: Literal["ac", "dc"] = Field(
        title="System",
        description="The bus (AC or DC) the block is connected to",
    )
    capex_spec: float = Field(
        title="Specific capital expenditures",
        description="Specific capital expenditures: cost in currency per installed peak power in W",
        ge=0,
    )
    mntex_spec: float = Field(
        title="Specific maintenance expenditures",
        description="Specific maintenance expenditures: cost in currency per year per installed peak power in W",
        ge=0,
    )
    opex_spec: float | str = Field(
        title="Specific operational expenditures",
        description="Specific operational expenditures: cost in currency per generated energy in Wh. Can be given as float or filename of a csv file containing a timeseries",
        json_schema_extra={"valid_values_or_format": "string with filename or [0, inf["},
    )
    ls: float = Field(
        title="Lifespan",
        description="Lifespan of the block in years after which it will be replaced",
        ge=1,
    )
    ccr: float = Field(
        title="Cost change ratio",
        description="Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan",
        ge=0,
    )
    eff_block: float = Field(
        title="Efficiency",
        description="Efficiency of the PV array, taking into account all losses occurring from insulation up to the point of feeding power into the bus to which the block is connected.",
        ge=0,
        le=1,
    )
    azimuth: float | str | None = Field(
        title="Surface azimuth",
        description="Clockwise from north (north=0, east=90, south=180, west=270). Ignored for tracking systems. Only considered if any API or 'Solcast file' is specified in data_source. None is equal to energy yield optimum.",
        json_schema_extra={"valid_values_or_format": "[0, 360[ or None."},
    )
    tilt: float | str | None = Field(
        title="Surface tilt angle",
        description="Tilt angle from horizontal plane. Ignored for two-axis tracking. Horizontal=0, Vertical=90. Only considered if any API or 'Solcast file' is specified in data_source. None sets the tilt angle to the specified location's latitude.",
        json_schema_extra={"valid_values_or_format": "[0, 90] or None"},
    )
    trackingtype: int | None = Field(
        title="Tracking type",
        description="Type of sun tracking. 0=fixed, 1=single horizontal axis aligned north-south, 2=two-axis tracking, 3=vertical axis tracking, 4=single horizontal axis aligned east-west, 5=single inclined axis aligned north-south. For data_source 'Solcast API' only 0 and 1 are valid. Ignored for any other data_source than 'PVGIS API' and 'Solcast API'.",
        json_schema_extra={"valid_values_or_format": "0, 1, 2, 3, 4, 5"},
    )
    horizon_custom: list[float] | None = Field(
        title="User horizon",
        description="Optional user specified elevation of horizon in degrees for 'PVGIS API', at equally spaced angular positions starting clockwise from north. Only valid if horizon is True. Not possible in combination with activated azimuth or tilt set to 'optimal'. Ignored for any other data_source than 'PVGIS API'.",
        json_schema_extra={
            "valid_values_or_format": 'list of floats (has to be specified surrounded by " ") e.g. "[45, 30, 0, 0]" or None'
        },
    )
    database: str | None = Field(
        title="Radiation database",
        description="Name of the radiation database for 'PVGIS-API'. Dependent on location and chosen simulation timeframe. 'PVGIS-SARAH' for Europe, Africa and Asia or 'PVGIS-NSRDB' for the Americas between 60°N and 20°S, 'PVGIS-ERA5' and 'PVGIS-COSMO' for Europe (including high-latitudes), and 'PVGIS-CMSAF' for Europe and Africa (will be deprecated).",
        json_schema_extra={
            "valid_values_or_format": "'PVGIS-SARAH2', 'PVGIS-SARAH3', 'PVGIS-NSRDB', 'PVGIS-ERA5', 'PVGIS-COSMO', 'PVGIS-CMSAF'"
        },
    )
    type_cell: str = Field(
        "Unkown",
        title="PV technology",
        description="PV technology for 'PVGIS API'.",
        json_schema_extra={"valid_values_or_format": "'crystSi', 'CIS', 'CdTe', 'Unknown'"},
    )
    mountingplace: Literal["free", "building"] = Field(
        title="Mounting place",
        description="Type of mounting for PV system for 'PVGIS API'. Options: free = free-standing, building = building-integrated.",
    )


class WindSourceModel(RevoletionBaseModel):
    """
    Wind turbine. Power potential is defined either in a csv timeseries file or retrieved from PVSource data containing wind speed, which is then converted to power for a specific turbine height.
    For the latter option, a PVSource block must exist.
    """

    _revoletion_docs_title: str = "WindSource"
    _revoletion_docs_icon: str = "💨"

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    size_preexisting_block: float = Field(
        title="Preexisting size",
        description="Installed rated power of wind turbine in W",
        ge=0,
    )
    capex_preexisting_block: bool = Field(
        title="Consideration of preexisting block size in capex",
        description="Trigger whether to consider existing component size in initial capex calculation. Replacement capex are unaffected.",
    )
    size_max_block: float | None = Field(
        title="Maximum size",
        description="Maximum size of WindSource including existing size. To enable unlimited investment set this parameter to None",
        ge=0,
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None"},
    )
    invest_block: bool = Field(
        title="Investment",
        description="Enable additional investment into the wind turbine",
    )
    system: Literal["ac", "dc"] = Field(
        title="System",
        description="The bus (AC or DC) the block is connected to",
    )
    data_source: str = Field(
        title="Data source",
        description="Data source for wind power. Wind power can either be given as a separate csv file or calculated from a PVSource block's data.",
        json_schema_extra={"valid_values_or_format": "'file' or a string with the name of a block of class PVSource"},
    )
    height: float = Field(
        title="Height",
        description="Hub height of the wind turbine in meters",
        ge=0,
    )
    filename: str | None = Field(
        title="Filename",
        description="Filename of csv file containing wind power data including the columns 'time' (timezone aware timestamps) and 'power_spec' (specific power in W per rated power in W). Only considered if 'file' is given in data_source.",
        json_schema_extra={"valid_values_or_format": "string with filename or None"},
    )
    capex_spec: float = Field(
        title="Specific capital expenditures",
        description="Specific capital expenditures: cost in currency per installed rated power in W",
        ge=0,
    )
    mntex_spec: float = Field(
        title="Specific maintenance expenditures",
        description="Specific maintenance expenditures: cost in currency per year per installed rated power in W",
        ge=0,
    )
    opex_spec: float | str = Field(
        title="Specific operational expenditures",
        description="Specific operational expenditures: cost in currency per generated energy in Wh. Can be given as float or filename of a csv file containing a timeseries.",
        json_schema_extra={"valid_values_or_format": "string with filename"},
    )
    ls: float = Field(
        title="Lifespan",
        description="Lifespan of the block in years after which it will be replaced",
        ge=1,
    )
    ccr: float = Field(
        title="Cost change ratio",
        description="Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan.",
        ge=0,
    )
    eff_block: float = Field(
        title="Efficiency",
        description="Efficiency of the wind turbine.",
        ge=0,
        le=1,
    )


class ControllableSourceModel(RevoletionBaseModel):
    """Independently controllable power sources (e.g. fossil generator, hydro power plant) that is unlimited in energy."""

    _revoletion_docs_title: str = "ControllableSource"
    _revoletion_docs_icon: str = " 🔩"

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    size_preexisting_block: float = Field(
        title="Preexisting size",
        description="Installed rated power of the source in W",
        ge=0,
    )
    capex_preexisting_block: bool = Field(
        title="Consideration of preexisting block size in capex",
        description="Trigger whether to consider preexisting component size specified in `size_preexisting_block` in initial capex calculation. Replacement capex are unaffected.",
    )
    size_max_block: float | None = Field(
        title="Maximum size",
        description="Maximum size of ControllableSource including preexisting size specified in size_preexisting_block. To enable unlimited investment set this parameter to None.",
        ge=0,
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None"},
    )
    invest_block: bool = Field(
        description="Enable additional investment into the power source",
        title="Investment",
    )
    system: Literal["ac", "dc"] = Field(description="The bus (AC or DC) the block is connected to")
    capex_spec: float = Field(
        title="Specific capital expenditures",
        description="Specific capital expenditures: cost in currency per installed power in W",
        ge=0,
    )
    mntex_spec: float = Field(
        title="Specific maintenance expenditures",
        description="Specific maintenance expenditures: cost in currency per year per installed peak power in W",
        ge=0,
    )
    opex_spec: float | str = Field(
        title="Specific operational expenditures",
        description="Specific operational expenditures: cost in currency per generated energy in Wh. Can be given as float or filename of a csv file containing a timeseries",
        json_schema_extra={"valid_values_or_format": "string with filename or [0, inf["},
    )
    ls: float = Field(
        title="Lifespan",
        description="Lifespan of the block in years after which it will be replaced",
        ge=1,
    )
    ccr: float = Field(
        title="Cost change ratio",
        description="Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan",
        ge=0,
    )
    eff_block: float = Field(
        title="Efficiency",
        description="Efficiency of the source",
        ge=0,
        le=1,
    )


class GridConnectionModel(RevoletionBaseModel):
    """Physical grid connection. A GridConnection instance requires one or multiple GridMarkets."""

    _revoletion_docs_title: str = "GridConnection"
    _revoletion_docs_icon: str = " 🔌⚡"

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    size_preexisting_g2s: float | Literal["equal"] = Field(
        title="Preexisting connection power from public grid to local site",
        description="Installed power for the power flow from the public grid to the local site (Grid2Site) in W. Set either `size_preexisting_g2s` or `size_preexisting_s2g` to 'equal' to set both directions' sizes to the same value.",
        json_schema_extra={"valid_values_or_format": "[0, inf[ or 'equal'"},
    )
    capex_preexisting_g2s: bool = Field(
        title="Consideration of preexisting AC/DC size in capex",
        description="Trigger whether to consider preexisting component size specified in size_preexisting_g2s in initial capex calculation. Replacement capex are unaffected.",
    )
    size_max_g2s: float | Literal["equal"] | None = Field(
        title="Maximum size of Grid2Site",
        description="Maximum size of Grid2Site including preexisting size specified in size_preexisting_g2s. To enable unlimited investment set this parameter to None. Set either size_max_g2s or size_max_s2g to 'equal' to set both directions' maximum investments to the same value.",
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None or 'equal'"},
    )
    invest_g2s: bool | Literal["equal"] = Field(
        title="Investment into Grid2Site",
        description="Enable additional investment into the maximum power from the grid to the local site. To ensure the same additional power for both directions set one invest variable to 'equal'.",
    )
    size_preexisting_s2g: float | Literal["equal"] = Field(
        title="Existing maximum power from local site to grid",
        description="Installed power for the power flow from the local site to the grid in W. To set both directions' existing powers to the same value set one size to 'equal'.",
        json_schema_extra={"valid_values_or_format": "[0, inf[ or 'equal'"},
    )
    capex_preexisting_s2g: bool = Field(
        title="Consider existing block size in capex",
        description="Trigger whether to consider existing component size in initial capex calculation. Replacement capex are unaffected.",
    )
    size_max_s2g: float | Literal["equal"] | None = Field(
        title="Maximum size of Site2Grid",
        description="Maximum size of Site2Grid including existing size. To enable unlimited investment set this parameter to None. To set both directions' maximum investments to the same value set one maximum investment to 'equal'.",
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None or 'equal'"},
    )
    invest_s2g: bool | Literal["equal"] = Field(
        title="Investment into Site2Grid",
        description="Enable additional investment into the maximum power from the local site to the grid. To ensure the same additional power for both directions set one invest variable to 'equal'.",
    )
    system: Literal["ac", "dc"] = Field(title="System", description="The bus (AC or DC) the block is connected to.")
    peakshaving: bool = Field(
        title="Activation of peak shaving",
        description="Trigger whether to consider peak power costs in the optimization (leads to peak shaving). Peak power costs will always be considered in the post-processing regardless the parameter specified here.",
    )
    peak_period: Literal["day", "week", "month", "year", "quarter"] = Field(
        title="Peak power cost period", description="Peak power cost period."
    )
    peak_power_init: float = Field(
        title="Initial peak power",
        description="Initial peak power per peak power period in W. Can be used in Rolling Horizon simulations to avoid overly reduced power consumption from the grid in first horizons of a peak period.",
        ge=0,
    )
    opex_spec_peak: float = Field(
        title="Specific operational expenditures for peak power",
        ge=0,
        description="Specific operational expenditures for maximum power drawn from the public grid per timestep in cost in currency per peak power in W per peak power period specified in peak_period. Resulting costs are always considered in post-processing, but are only taken into account by the optimizer, if peakshaving is set to 'True'.",
    )
    capex_spec: float = Field(
        title="Specific capital expenditures",
        ge=0,
        description="Specific capital expenditures: cost in currency per installed power (cumulative power of both directions) in W of the grid connection.",
    )
    mntex_spec: float = Field(
        title="Specific maintenance expenditures",
        ge=0,
        description="Specific maintenance expenditures: cost in currency per year per installed power (cumulative power of both directions) in W of the grid connection.",
    )
    ls: float = Field(
        title="Lifespan",
        ge=1,
        description="Lifespan of the block in years after which it will be replaced.",
    )
    ccr: float = Field(
        title="Cost change ratio",
        ge=0,
        description="Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan.",
    )
    eff_block: float = Field(title="Efficiency", ge=0, le=1, description="Efficiency of the grid connection.")
    markets: list[str] = Field(
        title="Markets",
        description="List containing names of GridMarket instances which are connected to the GridConnection.",
        json_schema_extra={"valid_values_or_format": "\"['name_of_market1', 'name_of_market2']\""},
    )


class GridMarketModel(RevoletionBaseModel):
    """Virtual GridMarket connected to a specific physical GridConnection."""

    _revoletion_docs_title: str = "GridMarket"
    _revoletion_docs_icon: str = " 📈"

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    res_only: bool = Field(
        title="Renewable energy sources only",
        description="If activated, selling energy to the grid is restricted to energy produced by renewable energies blocks (PVSource, WindSource) in the current timestep and energy stored in a storage with activated res_only parameter.",
    )

    opex_spec_g2s: str | float = Field(
        title="Specific operational expenditures for public grid to local site",
        description="Specific operational expenditures for buying energy: cost in currency per energy in Wh. Can be given as float or filename of a csv file containing a timeseries.",
        json_schema_extra={"valid_values_or_format": "string with filename or ]-inf, inf["},
    )
    opex_spec_s2g: str | float = Field(
        title="Specific operational expenditures for local site to public grid",
        description="Specific operational expenditures for selling energy: cost in currency per energy in Wh (set this parameter to a negative number to earn money for feeding in energy). Can be given as float or filename of a csv file containing a timeseries.",
        json_schema_extra={"valid_values_or_format": "string with filename or ]-inf, inf["},
    )
    pwr_s2g: float | None = Field(
        title="Power limit from public grid to local site",
        description="Power limit considered for the power flow from the public grid to the local site in W. If no additional limit for the market but only the limits of the physical grid connection should be taken into account, set to None.",
        ge=0,
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None"},
    )
    pwr_g2s: float | None = Field(
        title="Power limit from public grid to local site",
        description="Power limit considered for the power flow from the local site to the public grid in W. If no additional limit for the market but only the limits of the physical grid connection should be taken into account, set to None.",
        ge=0,
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None"},
    )


class StationaryBatteryModel(RevoletionBaseModel):
    """
    Stationary battery energy storage systems.
    A posteriori aging (i.e. capacity reduction) estimation is possible and will be taken into the next horizon as a reduced available SOC range.
    Storage modelling is done linearly without SOC or temperature based limits of charge or discharge power.
    """

    _revoletion_docs_title: str = "StationaryBattery"
    _revoletion_docs_icon: str = " 🔋"

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    size_preexisting_storage: float = Field(
        title="Preexisting size",
        description="Installed nominal capacity of the storage in Wh.",
        ge=0,
    )
    capex_preexisting_storage: bool = Field(
        title="Consideration of preexisting block size in capex",
        description="Trigger whether to consider preexisting component size specified in size_preexisting_storage in initial capex calculation. Replacement capex are unaffected.",
    )
    size_max_storage: float | None = Field(
        title="Maximum size",
        description="Maximum size of StationaryBattery including preexisting size specified in size_preexisting_storage. To enable unlimited investment set this parameter to None.",
        ge=0,
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None"},
    )
    invest_storage: bool = Field(
        title="Investment",
        description="Enable additional investment into the storage capacity.",
    )
    system: Literal["ac", "dc"] = Field(title="System", description="The bus (AC or DC) the block is connected to.")
    res_only: bool = Field(
        title="Renewable energy sources only",
        description="If activated, only energy from renewable sources (PVSource, WindSource) can be stored in the storage. This allows to feed energy from the storage into GridMarket instances with activated res_only parameter.",
    )
    aging: bool = Field(
        title="Consideration of battery aging",
        description="Battery aging calculation after each horizon. Aging results are taken into account for the next horizon by limiting the available SOC range. Maximum power is not reduced.",
    )
    chemistry: Literal["nmc", "lfp"] = Field(
        title="Cell Chemistry",
        description="Cell chemistry of the storage to select the correct aging model for aging calculation.",
    )
    temp_battery: float | str = Field(
        title="Battery temperature",
        description="Battery temperature used as stress factor in aging model. Can be set to a constant value, defined using the timeseries of a PVSource block as this contains a temperature timeseries, or set to None to inherit the temperature specified in temp_air of the Scenario.",
        json_schema_extra={
            "valid_values_or_format": "string with name of block of class StationaryBattery or ]-inf, inf["
        },
    )
    capex_spec: float = Field(
        title="Specific capital expenditures",
        description="Specific capital expenditures: cost in currency per installed nominal storage capacity in Wh.",
        ge=0,
    )
    mntex_spec: float = Field(
        title="Specific maintenance expenditures",
        description="Specific maintenance expenditures: cost in currency per year per installed nominal storage capacity in Wh.",
        ge=0,
    )
    opex_spec: float | str = Field(
        title="Specific operational expenditures",
        description="Specific operational expenditures: cost in currency per energy stored in the storage in Wh. Energy is measured at storage inflow. Can be given as float or filename of a csv file containing a timeseries.",
        json_schema_extra={"valid_values_or_format": "string with filename or [0, inf["},
    )
    ls: float = Field(
        title="Lifespan",
        description="Lifespan of the block in years after which it will be replaced.",
        ge=1,
    )
    eff_storage_roundtrip: float = Field(
        title="Roundtrip efficiency",
        description="Storage roundtrip efficiency. Charge and discharge efficiency is calculated using sqrt(eff_roundtrip).",
        ge=0,
        le=1,
    )
    eff_acdc: float = Field(
        title="Efficiency of the AC/DC converter",
        description="Efficiency of the AC/DC converter connecting the block to the AC bus. This parameter is neglected if the block is connected to the DC bus.",
        ge=0,
        le=1,
    )
    eff_dcac: float = Field(
        title="Efficiency of the DC/AC converter",
        description="Efficiency of the DC/AC converter connecting the block to the AC bus. This parameter is neglected if the block is connected to the DC bus.",
        ge=0,
        le=1,
    )
    crate_chg: float = Field(
        title="Charge C-rate",
        description="Maximum C-rate for charging.",
        ge=0,
    )
    crate_dis: float = Field(
        title="Discharge C-rate",
        description="Maximum C-rate for discharging.",
        ge=0,
    )
    soc_init: float = Field(
        title="Initial SOC",
        description="Initial SOC of the storage at simulation start.",
        ge=0,
        le=1,
    )
    q_loss_cal_init: float = Field(
        title="Initial capacity loss due to calendric aging",
        description="Initial capacity loss of the storage at simulation start due to calendric aging given as fraction of the total capacity. The capacity-related initial SOH is calculated by 1 - (q_loss_cal_init + q_loss_cyc_init).",
        ge=0,
        le=1,
    )
    q_loss_cyc_init: float = Field(
        title="Initial capacity loss due to cyclic aging",
        description="Initial cyclic loss of the storage at simulation start due to calendric aging given as fraction of the total capacity. The capacity-related initial SOH is calculated by 1 - (q_loss_cal_init + q_loss_cyc_init).",
        ge=0,
        le=1,
    )
    sdr: float = Field(
        title="Self discharge rate",
        description="Self discharge rate of storage components per month (30 days).",
        ge=0,
    )
    ccr: float = Field(
        title="Cost change ratio",
        description="Cost change ratio of the block's nominal price per year to be considered for replacement after its lifespan.",
        ge=0,
    )


class FleetModel(RevoletionBaseModel):
    """Fleet consisting of one or several SubFleets."""

    _revoletion_docs_title: str = "Fleet"
    _revoletion_docs_icon: str = " 🚚🚗"

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    system: Literal["ac", "dc"] = Field(
        title="System",
        description="The bus (AC or DC) the block is connected to",
    )
    subfleets: list[str] = Field(
        title="Subfleets",
        description="List of names of subfleets in Fleet in no particular order. Each of these subfleets must exist as such in the scenario file.",
    )
    data_source: Literal["usecases", "demand", "log"] = Field(
        title="Data source",
        description="Define whether usage timeseries (log file) should be (a) generated through mobility and dispatch simulation when given a usecase file, (b) generated through dispatch simulation only given a demand file or (c) read directly from a log file, forgoing a priori simulations.",
    )
    filename: str | None = Field(
        title="Filename of input file",
        description="Filename of csv file containing (a) usecase definition for DES, (b) sampled demand, or None, if the usage of a log file is specified in ```data_source```. Base search path is the scenario file's path, unless explicitly specified.",
        json_schema_extra={"valid_values_or_format": "string with filename or None"},
    )
    filename_mapper: str = Field(
        title="Filename of TimeframeMapper file",
        description="Filename of the file containing the mapping function assigning timeframes to individual days (e.g. weekday/weekend) for the Group's DES with or without the ending '.py'. The file itself has to be placed in the input directory. Base search path is the scenario file's path, unless explicitly specified.",
        json_schema_extra={"valid_values_or_format": "string with filename of python file with or without '.py'"},
    )
    pwr_lim_f2s: float | None = Field(
        ge=0.0,
        title="Power limit of fleet to site",
        description="Maximum power flow from Fleet to the local site (Fleet2Site) in W. To enable unlimited power flow set this parameter to None.",
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None"},
    )
    pwr_lim_s2f: float | None = Field(
        default=None,
        ge=0.0,
        title="Power limit of site to fleet",
        description="Maximum power flow from the local site to Fleet (Site2Fleet) in W. To enable unlimited power flow set this parameter to None.",
        json_schema_extra={"valid_values_or_format": "[0, inf[ or None"},
    )
    opex_spec_f2s: float | str = Field(
        title="Specific operational expenditures for fleet charging",
        description="Specific operational expenditures for Fleet charging: cost in currency per energy charged into Fleet in Wh. This can be used to simulate different operators for fleets and local energy grid. Negative costs can lead to unwanted behavior (e.g. wasting energy)! Can be given as float or filename of a csv file containing a timeseries",
        json_schema_extra={"valid_values_or_format": "string with filename or [0, inf["},
    )
    opex_spec_s2f: float | str = Field(
        title="Specific operational expenditures for fleet discharging",
        description="Specific operational expenditures for Fleet discharging: cost in currency per energy discharged from Fleet in Wh. This can be used to simulate different operators for fleets and local energy grid. Negative costs can lead to unwanted behavior (e.g. wasting energy)! Can be given as float or filename of a csv file containing a timeseries",
        json_schema_extra={"valid_values_or_format": "string with filename or [0, inf["},
    )


class SubFleetModel(RevoletionBaseModel):
    """
    SubFleet consisting of initially identical FleetUnits (Electric Vehicle, Internal Combustion Engine Vehicle, Mobile Battery).
    Behavior can either be given or generated within the integrated Discrete Event Simulation from stochastic behavioral parameters (use case definition in CSV file and python script with timeframe mapper)
    """

    _revoletion_docs_title: str = "SubFleet"
    _revoletion_docs_icon: str = " 🚗🚗"

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    """Pydantic model for Vehicle Commodity System configuration parameters."""

    num: int = Field(
        title="Number of fleet units",
        description="Number of fleet units within the SubFleet.",
        ge=1,
    )

    type_unit: Literal["ev", "icev", "mb"] = Field(
        title="Fleet unit type",
        description="Type of Fleet units contained in the Subfleet. 'ev': Electric Vehicle, 'icev': Internal Combustion Engine Vehicle, 'mb': Mobile Battery",
    )

    size_preexisting_storage: float = Field(
        ge=0.0,
        title="Preexisting size of storage",
        description="Installed nominal capacity of the Fleet unit's storage in Wh per single Fleet unit for all Fleet units within the Subfleet. Parameter is neglected if type_unit is set to 'icev'",
        json_schema_extra={"not_required_for": "`type_unit` == 'icev'"},
    )

    size_max_storage: float | None = Field(
        default=None,
        ge=0,
        title="Maximum size of storage",
        description="Maximum size of storage per Fleet unit including preexisting size specified in size_preexisting_storage. To enable unlimited investment set this parameter to None",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
            "valid_values_or_format": "[0, inf[ or None",
        },
    )

    invest_storage: bool = Field(
        title="Investment into storage",
        description="Enable additional investment into the Fleet units' storages",
        json_schema_extra={"not_required_for": "`type_unit` == 'icev'"},
    )

    capex_preexisting_storage: bool = Field(
        title="Consideration of preexisting block size in capex",
        description="Trigger whether to consider preexisting component size specified in size_preexisting_storage in initial capex calculation. Replacement capex are unaffected.",
    )

    capex_fix_glider: float = Field(
        ge=0.0,
        title="Capital expenditures for base vehicle",
        description="Fixed capital expenditures for each of the Fleet units in the SubFleet, irrespective of storage size, representing the base vehicle",
    )

    capex_preexisting_glider: bool = Field(
        title="Consideration of preexisting glider in capex",
        description="Trigger whether to consider preexisting glider capex specified in capex_fix_glider in initial capex calculation. Replacement capex are unaffected.",
    )

    capex_fix_charger: float = Field(
        ge=0.0,
        title="Capital expenditures for charger",
        description="Fixed capital expenditures for each of the Fleet units in the SubFleet, irrespective of storage size, representing the charger",
        json_schema_extra={"not_required_for": "`type_unit` == 'icev'"},
    )

    capex_preexisting_charger: bool = Field(
        title="Consideration of preexisting charger in capex",
        description="rigger whether to consider preexisting charger capex specified in capex_fix_charger in initial capex calculation. Replacement capex are unaffected.",
        json_schema_extra={"not_required_for": "`type_unit` == 'icev'"},
    )

    ccr: float = Field(
        ge=0.0,
        le=1.0,
        title="Cost change ratio",
        description="Cost change ratio of the block's (glider, storage, and charger) nominal price per year to be considered for replacement after its lifespan",
    )

    ls: float = Field(
        ge=1.0,
        title="Lifespan",
        description="Lifespan of the block (glider, storage, and charger) in years after which it will be replaced",
    )

    mntex_fix_glider: float = Field(
        ge=0.0,
        title="Fixed maintenance expenditures for glider",
        description="Fixed maintenance expenditures: cost in currency per year per Fleet unit, irrespective of traction battery size",
    )

    opex_spec_dist: float = Field(
        ge=0.0,
        title="Specific operational expenditures per distance",
        description="Specific operational expenditures per distance: cost in currency per driven distance in km",
    )

    crev_spec_time: float = Field(
        ge=0.0,
        title="Specific customer revenues per time",
        description="Specific customer revenues per time: Revenues from vehicle utilization specified as revenue in currency per used time in hours. Total revenue is calculated by summing up time and distance revenue",
    )

    crev_spec_dist: float = Field(
        ge=0.0,
        title="Specific customer revenues per distance",
        description="Specific customer revenues per distance: Revenues from vehicle utilization specified as revenue in currency per driven distance in km. Total revenue is calculated by summing up time and distance revenue",
    )

    opex_spec_ext_ac: float | str = Field(
        title="Specific operational expenditures for external AC charging",
        description="Specific operational expenditures for external AC charging: cost in currency per charged energy in Wh. Can be given as float or filename of a csv file containing a timeseries",
        ge=0,
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
            "valid_values_or_format": "string with filename or [0, inf[",
        },
    )

    opex_spec_ext_dc: float | str = Field(
        title="Specific operational expenditures for external DC charging",
        description="Specific operational expenditures for external DC charging: cost in currency per charged energy in Wh. Can be given as float or filename of a csv file containing a timeseries",
        ge=0,
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
            "valid_values_or_format": "string with filename or [0, inf[",
        },
    )

    capex_spec: float = Field(
        ge=0.0,
        title="Specific capital expenditures for storage",
        description="Specific capital expenditures for Fleet unit's storages: cost in currency per installed nominal storage capacity in Wh",
    )

    rex: str = Field(
        title="Range extender SubFleet",
        description="Name of a Mobile Battery SubFleet which can be used as Range Extender. Neglected, if (a) DES is not activated or (b) unit_type is not 'ev'",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
            "valid_values_or_format": "string with name of SubFleet",
        },
    )

    mode_scheduling: Literal["uc", "equal", "fcfs", "soc", "oc"] = Field(
        title="Scheduling Mode",
        description="Scheduling Mode for charging the SubFleet's Fleet units: Available options are uncoordinated charging ('uc'), three different rulebased strategies (equal distribution of the available power - 'equal', first come first served - 'fcfs', soc based charging - 'soc') and optimized charging ('oc'). Bidirectional charging is only available for 'oc'",
    )

    forecast_hours: float = Field(
        ge=0.0,
        title="Forecast hours",
        description="Neglected, if mode_scheduling is 'oc': Defines how much time in advance a trip can be seen by the charging scheduler in order to adjust the target SOC based on soc_target_high and soc_target_low. Feature currently not enabled",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    aging: bool = Field(
        title="Consideration of battery aging",
        description="Trigger whether to calculate battery aging for the Fleet unit's storage after each horizon. Aging results are taken into account for the next horizon by limiting the available SOC range. Maximum power is not reduced",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    chemistry: Literal["NMC", "LFP"] = Field(
        title="Cell chemistry",
        description="Cell chemistry of the storage to select the correct aging model for aging calculation",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    temp_battery: float | str = Field(
        title="Battery temperature",
        description="Battery temperature used as stress factor in aging model. Can be set to a constant value, defined using the timeseries of a PVSource block as this contains a temperature timeseries, or set to None to inherit the temperature specified in temp_air of the Scenario",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
            "valid_values_or_format": "string with name of block of class SubFleet or ]-inf, inf[",
        },
    )

    q_loss_cal_init: float = Field(
        ge=0.0,
        le=1.0,
        title="Initial capacity loss due to calendric aging",
        description="Initial capacity loss of the storage at simulation start due to calendric aging given as fraction of the total capacity. The capacity-related initial SOH is calculated by 1 - (q_loss_cal_init + q_loss_cyc_init)",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    q_loss_cyc_init: float = Field(
        ge=0.0,
        le=1.0,
        title="Initial capacity loss due to cyclic aging",
        description="Initial cyclic loss of the storage at simulation start due to cyclic aging given as fraction of the total capacity. The capacity-related initial SOH is calculated by 1 - (q_loss_cal_init + q_loss_cyc_init)",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    soc_init: float = Field(
        ge=0.0,
        le=1.0,
        title="Initial SOC",
        description="Initial SOC of the storage at simulation start",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    soc_target: float = Field(
        ge=0.0,
        le=1.0,
        title="Target SOC",
        description="Target SOC up to which DES charges a Fleet unit until it can be used for the next trip",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    soc_return: float = Field(
        ge=0.0,
        le=1.0,
        title="Return SOC",
        description="Minimum SOC after the trip. For DES this is used to calculate (a) the usable range of a battery and (b) the numbers of necessary Range Extender batteries if available",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    dsoc_buffer: float = Field(
        ge=0.0,
        le=1.0,
        title="Delta SOC buffer",
        description="Buffer given as SOC to compensate for aging occurring during the simulation and self discharge during a trip. Parameter is neglected for every other simulation paradigm than Rolling Horizon",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    pwr_chg_max: float = Field(
        ge=0.0,
        title="Maximum charging power",
        description="Maximum charging power at the local energy system in W",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    pwr_dis_max: float = Field(
        ge=0.0,
        title="Maximum discharging power",
        description="Maximum discharging power at the local energy system in W",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    pwr_ext_ac: float = Field(
        ge=0.0,
        title="Maximum power for external AC charging",
        description="Maximum charging power at an external AC charger in W",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    pwr_ext_dc: float = Field(
        ge=0.0,
        title="Maximum power for external DC charging",
        description="Maximum charging power at an external DC charger in W",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    eff_storage_roundtrip: float = Field(
        ge=0.0,
        le=1.0,
        title="Storage roundtrip efficiency",
        description="Roundtrip efficiency of the Fleet unit's storage measured at the connection between the Fleet unit's bus and the storage. Charge and discharge efficiency is calculated using sqrt(eff_storage_roundtrip)",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    eff_chg_ac: float = Field(
        ge=0.0,
        le=1.0,
        title="AC charging efficiency",
        description="Efficiency of the Fleet unit's On-Board-Charger (OBC) in charging direction. Taken into account for AC charging at the local grid (if Fleet's system is set to AC) and external AC charging",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    eff_chg_dc: float = Field(
        ge=0.0,
        le=1.0,
        title="DC charging efficiency",
        description="Efficiency of a DC charging station in charging direction. Taken into account for DC charging at the local grid only, as losses at public charging stations are not relevant for an fleet operator but the charging station operator only",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    eff_dis_ac: float = Field(
        ge=0.0,
        le=1.0,
        title="AC discharging efficiency",
        description="Efficiency of the commodity's On-Board-Charger (OBC) in discharging direction. Taken into account for DC charging at the local grid (if Fleet's system is set to AC) and external AC charging",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    eff_dis_dc: float = Field(
        ge=0.0,
        le=1.0,
        title="DC discharging efficiency",
        description="Efficiency of a DC charging station in discharging direction. Taken into account for DC discharging at the local grid only, as losses at public charging stations are not relevant for an fleet operator but the charging station operator only",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )

    sdr: float = Field(
        ge=0.0,
        title="Self discharge rate",
        description="Self discharge rate of storage component related to its nominal capacity per month (30 days)",
        json_schema_extra={
            "not_required_for": "`type_unit` == 'icev'",
        },
    )


# The block types which are currently being validate.
BLOCK_TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "FixedDemand": FixedDemandModel,
    "GridConnection": GridConnectionModel,
    "WindSource": WindSourceModel,
    "PVSource": PVSourceModel,
    "StationaryBattery": StationaryBatteryModel,
    "ControllableSource": ControllableSourceModel,
    "Fleet": FleetModel,
}


def validate_scenario_csv_file(scenario_path: Path) -> bool:
    """
    Validate the scenario definition CSV file at `scenario_path`.

    Args:
        scenario_path: Path to the scenario definitions file.

    Returns:
        Whether all scenario definitions are valid.
    """

    # Check if file exists
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario file not found at {scenario_path}")

    scenario_parameters = utils.read_scenario_from_file(scenario_path)

    for scenario_name in scenario_parameters.columns:
        if scenario_name.startswith("#"):
            # Scenarios starting with a '#' are excluded in `SimulationRun`.
            # This behavior is replicated there, to make sure only the intended scenarios are validated.
            _LOGGER.debug(f"Skipping validation of scenario '{scenario_name}'")
            continue

        # The blocks are mapped to python dicts to make processing with pydantic easier.
        blocks = defaultdict(dict)
        for (block, key), value in scenario_parameters[scenario_name].items():
            blocks[block][key] = value

        # The scenario config contains further information about which blocks are used, so it needs to be validated and loaded first.
        try:
            scenario_config = _validate_and_get_block("scenario", ScenarioModel, blocks)
        except RuntimeError as e:
            _LOGGER.warning(
                f"Failed to validate block 'scenario' in scenario {scenario_name} from {scenario_path}: {e}"
            )
            return False

        # The core block must always be specified, and it is not included in the blocks description in the scenario config.
        try:
            _ = _validate_and_get_block("core", SystemCoreModel, blocks)
        except RuntimeError as e:
            _LOGGER.warning(f"Failed to validate block 'core' in scenario {scenario_name} from {scenario_path}: {e}")
            return False

        # Now the other blocks defined in the scenario config can be validated.
        for block_name, block_type_name in scenario_config.blocks.items():
            if block_type_name not in BLOCK_TYPE_TO_MODEL:
                _LOGGER.debug(
                    f"Skipping block '{block_name}' with block type '{block_type_name}': No corresponding validation model is currently defined"
                )
                continue

            model = BLOCK_TYPE_TO_MODEL[block_type_name]
            try:
                _ = _validate_and_get_block(block_name, model, blocks)
            except RuntimeError as e:
                _LOGGER.error(
                    f"Failed to validate block '{block_name}' in scenario {scenario_name} from {scenario_path}: {e}"
                )
                return False

    return True


_RevoletionBaseModelT = TypeVar("_RevoletionBaseModelT", bound=RevoletionBaseModel)


def _validate_and_get_block(
    block_name: str,
    block_model_type: type[_RevoletionBaseModelT],
    blocks: dict[str, dict[str, Any]],
) -> _RevoletionBaseModelT:
    if block_name not in blocks:
        raise RuntimeError(f"Missing block '{block_name}'")

    block_dict = blocks[block_name]

    try:
        return block_model_type.model_validate(block_dict)
    except ValidationError as e:
        raise RuntimeError(str(e))
