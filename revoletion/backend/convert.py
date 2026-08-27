# TODO: This is a mainly AI-GENERATED helper lib, that should not live forever. Its only used to glue scientific and backend together without the need to refactor scientific heavily.

"""
Library that converts a REVOL-E-TION `ScenarioModel` back into the scenario
definition CSV format (columns: block, key, scenario).

This is the inverse of the scenario-to-blueprint mapping described in
`.opencode/skills/scenario-to-blueprint/SKILL.md`. The library is deliberately
agnostic of the concrete block set of a scenario: it contains no block names,
per-block key lists or key renames. Every non-bookkeeping field of every block
config is written verbatim, so any block type of the core repository (and
unknown ones) is handled without modification.

Row order is fully deterministic: the rows are sorted naturally
(alphabetic + numeric aware) by (block, key).

Usage:
    from scenario_to_csv import convert_to_csv

    path = convert_to_csv(scenario)  # scenario: revoletion_core.model.ScenarioModel
"""

from __future__ import annotations

import csv
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from revoletion_core.model import ScenarioModel

# Loosely typed JSON-ish value, as produced by ScenarioModel.model_dump().
type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


def _as_object_dict(value: JsonValue) -> dict[str, JsonValue]:
    """Narrow a loosely typed dump value to a JSON object (empty if it is not one)."""
    if isinstance(value, dict):
        return value
    return {}


# ---------------------------------------------------------------------------
# CSV dialect knowledge (not scenario-specific)
# ---------------------------------------------------------------------------

# BlockBaseModel bookkeeping fields that are never written to the CSV.
# Note: `bus` is inherited from BlockOnBusBaseModel and is redundant with the
# per-model `system` field; it is filtered out just in case it appears in a
# dump.
INTERNAL_FIELDS: frozenset[str] = frozenset({"name", "id", "enabled", "revoletion_model"})

# Blocks of these classes are hidden sub-blocks (GridMarketModel, referenced
# by grid.markets; SubFleetModel, referenced by fleet.subfleets) or the
# SystemCore. They never appear in the 'scenario,blocks' row but get their
# own row group in the CSV.
HIDDEN_BLOCK_MODELS: frozenset[str] = frozenset({"GridMarketModel", "SubFleetModel", "SystemCoreModel"})


# ---------------------------------------------------------------------------
# Value formatting (inverse of utils.infer_dtype)
# ---------------------------------------------------------------------------


def format_number(value: float) -> str:
    """Format a number such that infer_dtype() parses it back identically."""
    # Integral floats are written without a decimal point, matching the
    # original scenarios.csv style ('0' instead of '0.0').
    # Note: Non-integral floats are value-equivalent to the source but not
    # byte-identical (e.g. 1e-08 vs. the source spelling '1.00E-8').
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(value)


def format_starttime(value: JsonValue) -> str:
    """Write starttime in the compact 'd.m.Y' style used by scenarios.csv."""
    # Note: The forward mapping normalizes e.g. '1.1.2018' -> '01.01.2018
    # 00:00' because the raw value fails the %d.%m.%Y validator. We invert
    # that here: leading zeros are stripped and a trailing ' 00:00' is
    # dropped. Unparseable values pass through untouched. This is the only
    # field-specific transform in the library and can be removed if a fully
    # value-preserving round-trip is preferred.
    if not isinstance(value, str):
        return format_scalar(value)
    for fmt, with_time in (("%d.%m.%Y %H:%M", True), ("%d.%m.%Y", False)):
        try:
            # The tzinfo attachment only satisfies DTZ007; dt is read back
            # as plain calendar fields (day/month/year, %H:%M) below.
            dt = datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
        base = f"{dt.day}.{dt.month}.{dt.year}"
        if with_time and dt.strftime("%H:%M") != "00:00":
            base += f" {dt.strftime('%H:%M')}"
        return base
    return value


def unwrap_ref(value: JsonValue) -> JsonValue:
    """Unwrap a Ref (e.g. {'id': 'dem_timeseries'}) to its plain id."""

    if isinstance(value, dict) and set(value.keys()) == {"id", "type"}:
        identifier = value["id"]
        return identifier

    return value


def format_scalar(value: JsonValue) -> str:
    """Format a single value so that infer_dtype() round-trips it."""

    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return format_number(value)
    if isinstance(value, str):
        return value
    # Note: lists/dicts are written via repr() (single quotes), which is
    # exactly what the ast.literal_eval branch of infer_dtype() expects.

    return repr(value)


def format_value(value: JsonValue) -> str:
    """Format a config value, unwrapping Refs (incl. nested list refs)."""

    if isinstance(value, list):
        # subfleets -> [{'id': 'bev'}, ...], markets -> ['dynamic', ...]
        return format_scalar([unwrap_ref(item) for item in value])

    return format_scalar(unwrap_ref(value))


def format_blocks_dict(blocks: dict[str, str]) -> str:
    """Serialize ScenarioModel.blocks to the 'scenario,blocks' CSV cell."""
    # Only the "top-level" blocks appear in the CSV cell; hidden sub-blocks
    # (markets, subfleets) are excluded by class. Class names are written
    # without the 'Model' suffix, as in the reference scenarios.csv.
    visible = {name: model.removesuffix("Model") for name, model in blocks.items() if model not in HIDDEN_BLOCK_MODELS}
    return repr(visible)


# ---------------------------------------------------------------------------
# Row ordering
# ---------------------------------------------------------------------------


def natural_key(value: str) -> list[str | int]:
    """Split a string into chunks for an alphabetic + numeric sort."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", value)]


def row_sort_key(row: tuple[str, str]) -> tuple[list[str | int], list[str | int]]:
    return natural_key(row[0]), natural_key(row[1])


# ---------------------------------------------------------------------------
# ScenarioModel -> CSV rows
# ---------------------------------------------------------------------------


def _get_blocks(doc: dict[str, JsonValue]) -> dict[str, str]:
    """Return the blocks dict, deriving it from block_configs if absent."""
    if "blocks" in doc:
        return {str(name): str(model) for name, model in _as_object_dict(doc["blocks"]).items()}
    # Note: Fallback - derive the block order from block_configs when the
    # dump does not carry a top-level 'blocks' dict.
    derived: dict[str, str] = {}
    for name, cfg in _as_object_dict(doc.get("block_configs")).items():
        if name in {"context", "simulation_config", "system_core"}:
            continue
        derived[name] = str(_as_object_dict(cfg).get("revoletion_model", ""))
    return derived


def _pick(doc: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    """Pick a config dict, preferring the top-level over block_configs."""
    # Note: Dumps may repeat context/simulation_config/system_core inside
    # block_configs; the top-level copies are authoritative.
    top_level = _as_object_dict(doc.get(key))
    if top_level:
        return top_level
    return _as_object_dict(_as_object_dict(doc.get("block_configs")).get(key))


def collect_rows(doc: dict[str, JsonValue]) -> list[tuple[str, str, str]]:
    """Convert one ScenarioModel document into (block, key, value) rows."""
    rows: list[tuple[str, str, str]] = []

    context = _pick(doc, "context")
    sim_conf = _pick(doc, "simulation_config")
    core = _pick(doc, "system_core")
    block_configs = _as_object_dict(doc.get("block_configs"))

    # --- scenario rows: context and simulation_config plus the blocks cell.
    # The row order is irrelevant, the rows are sorted afterwards.
    for key, value in {**sim_conf, **context}.items():
        if key == "starttime":
            value = format_starttime(value)
        rows.append(("scenario", key, format_value(value)))
    rows.append(("scenario", "blocks", format_blocks_dict(_get_blocks(doc))))

    # --- core rows: all non-bookkeeping system_core fields.
    for key, value in core.items():
        if key not in INTERNAL_FIELDS:
            rows.append(("core", key, format_value(value)))

    # --- block rows: every non-bookkeeping field of every block config.
    # Block names and key lists are NOT hardcoded here, so any block type
    # present in the dump is handled. Legacy keys that the forward mapping
    # dropped are simply absent from the dump and produce no rows.
    for name, config in block_configs.items():
        if name in {"context", "simulation_config", "system_core"}:
            continue
        for key, value in _as_object_dict(config).items():
            if key in INTERNAL_FIELDS:
                continue

            if key == "bus":
                rows.append((name, "system", format_value(value)))
            else:
                rows.append((name, key, format_value(value)))

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_to_csv(scenario: ScenarioModel, parent_dir: Path) -> Path:
    """Convert a ScenarioModel into a CSV file and return its path.

    The CSV is written to a temporary file in the system temp directory.
    Note: The file is created with delete=False because the caller needs the
    path after this function returns; remove it when it is no longer needed.
    """
    doc: dict[str, JsonValue] = scenario.model_dump()

    ordered: list[tuple[str, str]] = []
    columns: dict[tuple[str, str], str] = {}
    for block, key, value in collect_rows(doc):
        pos = (block, key)
        if pos not in columns:
            ordered.append(pos)
            columns[pos] = value
    ordered.sort(key=row_sort_key)

    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        suffix=".csv",
        prefix="tmp_scenario_",
        delete=False,
        dir=parent_dir,
    ) as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["block", "key", "scenario"])
        for block, key in ordered:
            writer.writerow([block, key, columns[(block, key)]])
        path = Path(fh.name)

    return path
