"""
REVOL-E-TION Model Schema Documentation Generator

This script automatically generates Markdown-formatted documentation tables from the REVOL-E-TION pydantic models and injects them into the README.md file.

Usage:
    python scripts/generate_scenario_schema.py

The script will automatically locate the README.md file in the parent directory and
update it with generated documentation between the defined entry and exit point markers.
"""

import textwrap
from pathlib import Path
from typing import Any

from revoletion.models import (
    ControllableSourceModel,
    FixedDemandModel,
    FleetModel,
    GridConnectionModel,
    GridMarketModel,
    PVSourceModel,
    RevoletionBaseModel,
    ScenarioModel,
    StationaryBatteryModel,
    SubFleetModel,
    SystemCoreModel,
    WindSourceModel,
)

# Matching entry point markers are placed in the README.md to automatically select the right location for the schema tables.
ENTRY_POINT_SCENARIO_SCHEMA_TABLES = "<!ENTRY_POINT_SCHEMA_TABLES>"
EXIT_POINT_SCENARIO_SCHEMA_TABLES = "<!EXIT_POINT_SCHEMA_TABLES>"

BLOCK_HTML_TEMPLATE = """
<details style="margin-bottom: 1em;">
<summary style="border: 2px solid #333333;
  padding: 10px;
  font-weight: bold;
  border-radius: 6px;
  cursor: pointer;
">
  <span style="display: inline-block;width: 3em;text-align: center;font-size: 1.5em;">
    {block_icon}
  </span>
  {block_title}
</summary>

{block_description}

| Key | Name | Type | Not required for | Description | Valid values or format |
|-----|------|------|------------------|-------------|------------------------|
{table_content}

</details>
"""

# The models for which the documentation should be generated.
MODELS = [
    ScenarioModel,
    SystemCoreModel,
    FixedDemandModel,
    PVSourceModel,
    WindSourceModel,
    ControllableSourceModel,
    GridConnectionModel,
    GridMarketModel,
    StationaryBatteryModel,
    FleetModel,
    SubFleetModel,
]


def _parse_bounds(field: dict[str, str | float | int]) -> str:
    """
    Parse the bounds of a pydantic field into a human-readable format.
    """
    bounds = []
    if "exclusiveMinimum" in field:
        bounds.append(f"]{field['exclusiveMinimum']}")
    elif "minimum" in field:
        bounds.append(f"[{field['minimum']}")
    else:
        # No lower bound specified, by default the lower bound is negative infinity.
        bounds.append("]-inf")

    if "exclusiveMaximum" in field:
        bounds.append(f"{field['exclusiveMaximum']}[")
    elif "maximum" in field:
        bounds.append(f"{field['maximum']}]")
    else:
        # No upper bound specified, by default the upper bound is positive infinity.
        bounds.append("inf[")

    return ", ".join(bounds)


def _resolve_field_types(field: dict[str, Any]) -> list[str]:
    """Recursively resolve which types are valid for one field."""
    if "anyOf" in field:
        field_types = []
        # `anyOf` relates to union types (e.g. `float | None`)
        for sub_field in field["anyOf"]:
            if "type" not in sub_field:
                # For some combinations, the JSON schema contains nested `anyOf` sections.
                # Those are resolved recursively here.
                field_types.extend(_resolve_field_types(sub_field))
            else:
                field_types.append(sub_field["type"])
        return field_types
    elif "type" in field:
        return [field["type"]]
    else:
        return []


_TYPE_MAP = {
    "null": "None",
    "integer": "int",
    "string": "str",
    "number": "float",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
}


def generate_markdown_table_for_block(model: type[RevoletionBaseModel]) -> str:
    """
    Generate Markdown documentation with table for a single REVOL-E-TION model.

    Extracts field information from the model's JSON schema and formats it into
    an Markdown table within a collapsible details block.

    Args:
        model: Pydantic model class to generate documentation for

    Returns:
        str: Complete Markdown block containing formatted documentation table
    """
    rows = []

    # Use the JSON schema as intermediate, as it already handles the type resolution and
    # reduces the dependency on pydantic.
    schema = model.model_json_schema()
    for field_name, field in schema["properties"].items():
        # Get field details
        field_title = field.get("title", field_name.replace("_", " ").title())

        field_types = _resolve_field_types(field)

        field_desc = field.get("description", "")

        # Determine valid values/format
        # NOTE: Currently, the automatic bounds generation is not supported for union types with None.
        # For those fields the bounds must be specified in `valid_values_or_format` in `json_schema_extra`.
        valid_values = []
        if "valid_values_or_format" in field:
            # The field has a specific `valid_values_or_format` set which takes precedence.
            valid_values.append(field["valid_values_or_format"])
        else:
            if "enum" in field:
                valid_values.append(", ".join([f"'{v}'" for v in field["enum"]]))

            if "integer" in field_types:
                valid_values.append(_parse_bounds(field))

            if "number" in field_types:
                valid_values.append(_parse_bounds(field))

            if "boolean" in field_types:
                valid_values.append("True, False")

            if "null" in field_types:
                valid_values.append("None")

        valid_values_or_format_str = " or ".join(valid_values)

        # Specified in `json_schema_extra`.
        not_required_for = field.get("not_required_for") or ""

        if "examples" in field:
            valid_values_or_format_str += f" Examples: {' or '.join(field['examples'])}"

        field_type_str = " or ".join(_TYPE_MAP[field_type] for field_type in field_types)

        rows.append(
            f"| `{field_name}` | {field_title} | {field_type_str} | {not_required_for} | {field_desc} | {valid_values_or_format_str} |"
        )

    table_content = "\n".join(rows)
    # Get special attributes, which should be defined on each model.
    block_description = ""
    if model.__doc__ is not None:
        block_description = textwrap.dedent(model.__doc__).strip()
    block_title = model._revoletion_docs_title.default
    block_icon = model._revoletion_docs_icon.default

    formatted_block = BLOCK_HTML_TEMPLATE.format(
        block_title=block_title,
        block_icon=block_icon,
        block_description=block_description,
        table_content=table_content,
    )

    return formatted_block


def inject_markdown_tables_into_file(file_path: Path) -> None:
    """
    Inject generated documentation into a file between the specified markers.

    Reads the target file, locates the entry and exit point markers, and replaces
    the content between them with newly generated documentation tables for all
    configured REVOL-E-TION models.

    Args:
        file_path: Path to the file where documentation should be injected, e.g., the README.md.
    """
    file_content = file_path.read_text(encoding="utf-8")
    content_before_schema_table = file_content.split(ENTRY_POINT_SCENARIO_SCHEMA_TABLES)[0]
    content_after_schema_table = file_content.split(EXIT_POINT_SCENARIO_SCHEMA_TABLES)[1]

    formatted_blocks = []
    for model in MODELS:
        formatted_blocks.append(generate_markdown_table_for_block(model))

    schema_tables = "\n".join(formatted_blocks)

    new_file_content = "\n".join(
        [
            content_before_schema_table.rstrip(),
            "\n",
            ENTRY_POINT_SCENARIO_SCHEMA_TABLES,
            schema_tables,
            EXIT_POINT_SCENARIO_SCHEMA_TABLES,
            "\n",
            content_after_schema_table.lstrip(),
        ]
    )

    file_path.write_text(new_file_content, encoding="utf-8")


if __name__ == "__main__":
    readme_path = Path(__file__).parents[1] / "README.md"
    inject_markdown_tables_into_file(readme_path)
