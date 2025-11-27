from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import pandas as pd
import plotly.graph_objects as go
from typing_extensions import override

from revoletion import utils

from . import blocks

_BlockRegistryT = dict[str, Any]

_R = TypeVar("_R")


class BlockVisitor(Generic[_R], ABC):
    @abstractmethod
    def visit_block(self, block: blocks.BaseBlock, *args, **kwargs) -> _R: ...


class PlotTraces:
    _plot_traces: list[go.Scatter]
    _secondary_y: list[bool]

    def __init__(self) -> None:
        self._plot_traces = []
        self._secondary_y = []

    def append(self, plot_line: go.Scatter, secondary_y: bool = False) -> None:
        self._plot_traces.append(plot_line)
        self._secondary_y.append(secondary_y)

    def extend(self, plot_lines: list[go.Scatter], secondary_ys: list[bool] | None = None) -> None:
        if not secondary_ys:
            secondary_ys = [False] * len(plot_lines)

        self._plot_traces.extend(plot_lines)
        self._secondary_y.extend(secondary_ys)

    @property
    def plot_lines(self) -> list[go.Scatter]:
        return self._plot_traces

    @property
    def secondary_ys(self) -> list[bool]:
        return self._secondary_y


class VisualizationBlockVisitor(BlockVisitor[None]):
    """
    Visitor to create the plot traces after a run.
    """

    def create_plot_traces(self, block_registry: _BlockRegistryT) -> PlotTraces:
        plot_traces = PlotTraces()
        for block in block_registry.get("TopLevelBlock", {}).values():
            self.visit_block(block, plot_traces=plot_traces)
        return plot_traces

    @override
    def visit_block(self, block: blocks.BaseBlock, plot_traces: PlotTraces) -> None:
        if isinstance(block, blocks.NonElectricBlock):
            # Abort for non-electric blocks.
            return

        for subblock in block.subblocks.values():
            self.visit_block(subblock, plot_traces=plot_traces)

        # The system core block is special. It is an electric block, but does not employ the same
        # plotting logic. Therefore, only the custom plotting logic is executed and the execution
        # does not fall through like for the other block types.
        if isinstance(block, blocks.SystemCore):
            return self.visit_system_core(block, plot_traces=plot_traces)

        if isinstance(block, blocks.ElectricBlock):
            self.visit_electric_block(block, plot_traces=plot_traces)

        if isinstance(block, blocks.RenewableSource):
            self.visit_renewable_source(block, plot_traces=plot_traces)

        if isinstance(block, blocks.StorageBlock):
            self.visit_storage_block(block, plot_traces=plot_traces)

        if isinstance(block, blocks.ElectricFleetUnit):
            self.visit_electric_fleet_unit(block, plot_traces=plot_traces)

    def visit_electric_block(self, block: blocks.ElectricBlock, plot_traces: PlotTraces) -> None:
        plot_traces.append(
            plot_line=go.Scatter(
                x=block.scenario.times.eval.dti,
                y=block.flows.loc[block.scenario.times.eval.dti, "total"],
                mode="lines",
                name=self.get_legend_entry(block),
                line=dict(width=2, dash=None, shape="hv"),
                visible=True if block.top_level_block else "legendonly",
            ),
            secondary_y=False,
        )

    def visit_system_core(self, block: blocks.SystemCore, plot_traces: PlotTraces) -> None:
        plot_traces.extend(
            plot_lines=[
                go.Scatter(
                    x=block.scenario.times.eval.dti,
                    y=block.flows.loc[block.scenario.times.eval.dti, "dcac"],
                    mode="lines",
                    name=f"{block.name} DC-AC power (max. {block.sizes['dcac'].total / 1e3:.1f} kW)",
                    line=dict(width=2, dash=None, shape="hv"),
                    visible="legendonly",
                ),
                go.Scatter(
                    x=block.scenario.times.eval.dti,
                    y=block.flows.loc[block.scenario.times.eval.dti, "acdc"],
                    mode="lines",
                    name=f"{block.name} AC-DC power (max. {block.sizes['acdc'].total / 1e3:.1f} kW)",
                    line=dict(width=2, dash=None, shape="hv"),
                    visible="legendonly",
                ),
            ],
            secondary_ys=[False, False],
        )

    def visit_renewable_source(self, block: blocks.RenewableSource, plot_traces: PlotTraces) -> None:
        plot_traces.extend(
            plot_lines=[
                go.Scatter(
                    x=block.scenario.times.eval.dti,
                    y=-1 * block.flows.loc[block.scenario.times.eval.dti, "curt"],
                    mode="lines",
                    name=f"{block.name} curtailed power",
                    line=dict(width=2, dash=None, shape="hv"),
                    visible="legendonly",
                ),
                go.Scatter(
                    x=block.scenario.times.eval.dti,
                    y=block.flows.loc[block.scenario.times.eval.dti, "pot"],
                    mode="lines",
                    name=f"{block.name} potential power",
                    line=dict(width=2, dash=None, shape="hv"),
                    visible="legendonly",
                ),
            ],
            secondary_ys=[False, False],
        )

    def visit_storage_block(self, block: blocks.StorageBlock, plot_traces: PlotTraces) -> None:
        data_soc = block.states.loc[block.scenario.times.eval.dti_extd, "soc"].dropna()
        data_soh = block.states.loc[block.scenario.times.eval.dti_extd, "soh"].dropna()
        plot_traces.extend(
            plot_lines=[
                go.Scatter(
                    x=data_soc.index,
                    y=data_soc,
                    mode="lines",
                    name=f"{block.name} SOC",
                    line=dict(width=2, dash=None),
                    visible="legendonly",
                ),
                go.Scatter(
                    x=data_soh.index,
                    y=data_soh,
                    mode="lines",
                    name=f"{block.name} SOH",
                    line=dict(width=2, dash=None),
                    visible="legendonly",
                ),
            ],
            secondary_ys=[True, True],
        )

    def visit_electric_fleet_unit(self, block: blocks.ElectricFleetUnit, plot_traces: PlotTraces) -> None:
        legend = f"{block.name} consumption power"
        plot_traces.append(
            plot_line=go.Scatter(
                x=block.scenario.times.eval.dti,
                y=block.log.loc[block.scenario.times.eval.dti, "consumption"],
                mode="lines",
                name=legend,
                line=dict(width=2, dash=None, shape="hv"),
                visible="legendonly",
            ),
            secondary_y=False,
        )

        for mode in ["ac", "dc"]:
            pwr = getattr(self, f"pwr_ext_{mode}_max", 0)
            if pwr == 0:
                continue

            legend = f"{block.name} external {mode.upper()} charging power (max. {pwr / 1e3:.1f} kW)"
            plot_traces.append(
                plot_line=go.Scatter(
                    x=block.scenario.times.eval.dti,
                    y=block.flows.loc[block.scenario.times.eval.dti, f"ext_{mode}"],
                    mode="lines",
                    name=legend,
                    line=dict(width=2, dash=None, shape="hv"),
                    visible="legendonly",
                ),
                secondary_y=False,
            )

    def get_legend_entry(self, block: blocks.BaseBlock) -> str:
        match block:
            case blocks.RenewableSource():
                return f"{block.name} power (nom. {block.sizes['block'].total / 1e3:.1f} kW)"
            case blocks.FixedDemand():
                return f"{block.name} power"
            case blocks.GridConnection():
                return (
                    f"{block.name} power (max. {block.sizes['g2s'].total / 1e3:.1f} kW from / "
                    f"{block.sizes['s2g'].total / 1e3:.1f} kW to grid)"
                )
            case blocks.GridMarket():
                powers = {
                    power: min(
                        block.parent.sizes[power].total,
                        (
                            getattr(block, f"pwr_{power}")
                            if pd.notna(getattr(block, f"pwr_{power}"))
                            else block.parent.sizes[power].total
                        ),
                    )
                    for power in ["g2s", "s2g"]
                }

                return f"{block.name} power (max. {powers['g2s'] / 1e3:.1f} kW from / {powers['s2g'] / 1e3:.1f} kW to grid)"
            case blocks.StationaryBattery():
                return (
                    f"{block.name} (dis-)charge power "
                    f"(max. {block.sizes['storage'].total * block.crate_chg * block.eff['chg'] / 1e3:.1f} kW charge / "
                    f"{block.sizes['storage'].total * block.crate_dis * block.eff['dis'] / 1e3:.1f} kW discharge)"
                )

            case blocks.Fleet():

                def lim2str(lim) -> str:
                    if lim is None:
                        return "unlimited"
                    else:
                        return f"{lim / 1e3:.1f}"

                return (
                    f"{block.name} power (max. {lim2str(block.pwr_lim_s2f)} kW charge / "
                    f"{lim2str(block.pwr_lim_f2s)} kW discharge)"
                )
            case blocks.ElectricFleetUnit():
                return (
                    f"{block.name} (dis-)charge power "
                    f"(max. {block.pwr_chg_max / 1e3:.1f} kW charge / "
                    f"{(block.pwr_dis_max * block.eff['dis_int']) / 1e3:.1f} kW discharge)"
                )

            case _:
                return f"{block.name} power (max. {block.sizes['block'].total / 1e3:.1f} kW)"


class MessageCollectionBlockVisitor(BlockVisitor[list[str]]):
    def collect_messages(self, block_registry: _BlockRegistryT) -> list[str]:
        messages = []
        for block in block_registry.get("TopLevelBlock", {}).values():
            messages.extend(self.visit_block(block))
        return messages

    @override
    def visit_block(self, block: blocks.BaseBlock) -> list[str]:
        messages = []
        for subblock in block.subblocks.values():
            messages.extend(self.visit_block(subblock))

        for size in block.sizes.values():
            if (msg := size.result_msg) != "":
                messages.append(msg)

        if isinstance(block, blocks.GridConnection):
            messages.extend(self.visit_grid_connection(block))

        return messages

    def visit_grid_connection(self, block: blocks.GridConnection) -> list[str]:
        return [
            f'{"Optimized peak" if block.peakshaving else "Peak"} power in component "{block.name}" for peak period '
            f'"{period}": {row["power"] / 1e3:.1f} kW '
            f"- OPEX in simulation period: {block.evaluators[period].opex_peak.sim:.2f} {block.scenario.currency}"
            for period, row in block.peak_periods.iterrows()
            if row["start"] < block.scenario.times.eval.end
        ]


class TimeseriesCollectionBlockVisitor(BlockVisitor[list[pd.Series]]):
    def collect_timeseries(self, block_registry: _BlockRegistryT, horizon: utils.TimeSettings) -> list[pd.Series]:
        timeseries = []
        for block in block_registry.get("TopLevelBlock", {}).values():
            timeseries.extend(self.visit_block(block, horizon))
        return timeseries

    @override
    def visit_block(self, block: blocks.BaseBlock, horizon: utils.TimeSettings) -> list[pd.Series]:
        timeseries = []
        for subblock in block.subblocks.values():
            timeseries.extend(self.visit_block(subblock, horizon))

        if isinstance(block, blocks.ElectricBlock):
            timeseries.extend(self.visit_electric_block(block, horizon))

        return timeseries

    def visit_electric_block(self, block: blocks.ElectricBlock, horizon: utils.TimeSettings) -> list[pd.Series]:
        timeseries = []
        reindexed_flows = block.flows.copy()
        reindexed_flows.columns = pd.MultiIndex.from_tuples(
            tuples=[(block.name, col) for col in reindexed_flows.columns],
            names=["block", "key"],
        )
        timeseries.append(reindexed_flows.loc[horizon.dti, :])

        reindexed_states = block.states.copy()
        reindexed_states.columns = pd.MultiIndex.from_tuples(
            tuples=[(block.name, col) for col in reindexed_states.columns],
            names=["block", "key"],
        )
        timeseries.append(reindexed_states.loc[horizon.dti_extd, :])
        return timeseries


class SummaryCollectionBlockVisitor(BlockVisitor[list[pd.DataFrame]]):
    def collect_summary(self, block_registry: _BlockRegistryT) -> list[pd.DataFrame]:
        summary_df_list = []
        for block in block_registry.get("TopLevelBlock", {}).values():
            block_summary_df_list = self.visit_block(block)
            summary_df_list.extend(block_summary_df_list)
        return summary_df_list

    def _convert_summary_list_to_dataframe(self, block_name: str, summary_list: list[pd.Series]) -> pd.DataFrame:
        summary_df = pd.concat(summary_list)
        summary_df.index = pd.MultiIndex.from_tuples(
            tuples=[(block_name, key) for key in summary_df.index],
            names=["block", "key"],
        )
        return summary_df

    @override
    def visit_block(self, block: blocks.BaseBlock) -> list[pd.DataFrame]:
        summary_df_list = []

        # For each subblock the dataframe is just passed through.
        # This is necessary, because the results for each block should be aggregated separatly.
        for subblock in block.subblocks.values():
            summary_df_list.extend(self.visit_block(subblock))

        # To construct the dataframe for a block, we first collect the individual series into a list.
        summary_list = []
        # get attributes of type int, float, bool and str for scenario.summary_list
        summary_list.append(
            pd.Series(
                {key: value for key, value in block.__dict__.items() if isinstance(value, (int, float, bool, str))}
            )
        )

        # get energy results
        for size in block.sizes.values():
            summary_list.append(size.result_summary)

        # get economic results
        summary_list.append(block.aggregator.write_result_summary())

        if isinstance(block, blocks.ElectricBlock):
            summary_list.append(self.visit_electric_block(block))

        if isinstance(block, blocks.GridConnection):
            summary_list.append(self.visit_grid_connection(block))

        # After all individual summary series have been collected, they can be aggregated into the final dataframe.
        summary_df_list.append(self._convert_summary_list_to_dataframe(block.name, summary_list))

        return summary_df_list

    def visit_electric_block(self, block: blocks.ElectricBlock) -> pd.Series:
        return utils.create_results_from_dataframe(df=block.energies, name_prefix="energy")

    def visit_grid_connection(self, block: blocks.GridConnection) -> pd.Series:
        peak_power_results = {}
        for period, row in block.peak_periods.iterrows():
            if row["start"] < block.scenario.times.eval.end:
                peak_power_results.update(
                    {
                        f"{period}_peak_power": row["power"],
                        f"{period}_peak_period_fraction": row["period_fraction"],
                        f"{period}_peak_opex_sim": block.evaluators[period].opex_peak.sim,
                    }
                )
        return pd.Series(peak_power_results)
