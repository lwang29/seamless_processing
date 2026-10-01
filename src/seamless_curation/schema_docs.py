"""Render ``docs/annotation_schema.md`` from the registry.

The field reference is generated, never hand-edited, so it cannot disagree with
the tables: ``tests/test_schema.py`` fails when the committed file is stale.
Regenerate with ``python -m seamless_curation --config ... schema-docs``.
"""

from __future__ import annotations

from . import schema

INTRO = """# Annotation schema — field reference

*Generated from `src/seamless_curation/schema.py`; do not edit by hand.*

Eight tables, one per level. Every column lives on exactly one table; only keys
(`id`, `link`, `id_part`, `group_key`) and the partition columns (`vendor`,
`label`, `split`) repeat, and partitions always equal the parent row's value.
Higher-level summaries of clip concepts carry a level prefix
(`recording_posture` vs the clip's `posture`), so plain joins never collide.

| table | level | primary key | rows (annotations_v1) |
|---|---|---|---|
{table_rows}

Column attributes:

* **role** — `id` primary key; `link` foreign key; `id_part` raw Meta id
  segment; `group_key` grouping key; `partition` vendor/label/split; `metadata`
  given attribute; `measure` continuous observation; `label` categorical
  annotation; `score` 0-1 annotation; `confidence` 0-1 support for a label or
  score; `status` why a value is or isn't present; `flag` non-null boolean.
* **provenance** — `meta:as_is` Meta's value unchanged; `meta:derived` computed
  by us from Meta's metadata/features; `prior:reused` a definition from the
  previous (filtering) pipeline reused unchanged and recomputed over every clip;
  `prior:adapted` a previous definition or threshold, changed (see the note);
  `fresh` new.
* **evidence** — `given` supplied by Meta; `parsed` decoded from an id or code;
  `measured` computed from released signals (SMPL-H and FAU are themselves model
  estimates); `rule_inferred` a category/flag from a stated rule on
  measurements; `aggregate` a summary of lower-level rows.
* **NA means** — when a column may be missing, exactly what a missing value
  means. A missing value never stands for zero.
"""

ROW_COUNTS: dict[str, str] = {}


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render(row_counts: dict[str, int] | None = None) -> str:
    counts = row_counts or {}
    table_rows = "\n".join(
        f"| `{t}` | {schema.LEVELS[t]} | `{schema.PRIMARY_KEYS[t]}` | "
        f"{counts[t]:,} |" if t in counts else
        f"| `{t}` | {schema.LEVELS[t]} | `{schema.PRIMARY_KEYS[t]}` | see annotation_summary.json |"
        for t in schema.TABLES
    )
    parts = [INTRO.format(table_rows=table_rows)]
    for table, fields in schema.iter_tables():
        parts.append(f"\n## `{table}` — {schema.LEVELS[table]} level\n")
        parts.append("| column | type | role | unit | allowed values | NA means | provenance | evidence | description |")
        parts.append("|---|---|---|---|---|---|---|---|---|")
        for f in fields:
            allowed = ", ".join(f"`{a}`" for a in f.allowed) if f.allowed else ""
            description = f.description + (f" *Provenance note:* {f.note}" if f.note else "")
            if f.qualified_by:
                description += f" *Read with:* {', '.join(f'`{q}`' for q in f.qualified_by)}."
            parts.append(
                f"| `{f.name}` | {f.dtype} | {f.role} | {_cell(f.unit)} | {_cell(allowed)} | "
                f"{_cell(f.missing) if f.nullable else 'never NA'} | {f.provenance} | {f.evidence} | "
                f"{_cell(description)} |"
            )
    return "\n".join(parts) + "\n"
