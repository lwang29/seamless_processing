"""The registry is the specification: it must be self-consistent and match its docs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seamless_curation import schema
from seamless_curation.schema_docs import render

REPO = Path(__file__).resolve().parents[1]


def test_the_registry_is_internally_consistent() -> None:
    assert schema.registry_problems() == []


def test_every_table_has_its_primary_key_first() -> None:
    for table, key in schema.PRIMARY_KEYS.items():
        assert schema.columns(table)[0] == key


def test_only_keys_and_partitions_repeat_across_tables() -> None:
    seen: dict[str, set[str]] = {}
    for f in schema.fields():
        seen.setdefault(f.name, set()).add(f.role)
    for name, roles in seen.items():
        tables = [f.table for f in schema.fields() if f.name == name]
        if len(tables) > 1:
            assert roles <= set(schema.SHARED_ROLES), (name, tables, roles)


def test_interaction_id_is_never_published() -> None:
    """It is a prompt id; grouping conversations by it merged 1,550 sessions."""

    assert not any(f.name == "interaction_id" for f in schema.fields())


def test_flags_are_non_null_bools() -> None:
    for f in schema.fields():
        if f.role == "flag":
            assert f.dtype == "bool" and not f.nullable, f.name


def test_the_committed_field_reference_is_current() -> None:
    committed = (REPO / "docs" / "annotation_schema.md").read_text(encoding="utf-8")
    assert committed == render(), "run: python -m seamless_curation schema-docs"


def test_conform_orders_casts_and_refuses_gaps() -> None:
    frame = pd.DataFrame({name: [None] for name in schema.columns("interaction_windows")})
    with pytest.raises(ValueError):
        schema.conform(frame, "interaction_windows")  # non-nullable window_id is NA
    frame["window_id"] = ["V00_S0001_I00000001_L30_W000"]
    frame["interaction_key"] = ["V00_S0001_I00000001"]
    frame["vendor"], frame["label"], frame["split"] = "V00", "improvised", "train"
    frame["window_index"] = [0]
    frame["window_start_s"], frame["window_end_s"] = [0.0], [30.0]
    frame["window_posture_pair"] = ["standing+standing"]
    out = schema.conform(frame, "interaction_windows")
    assert list(out.columns) == schema.columns("interaction_windows")
    assert str(out["window_turn_switches"].dtype) == "Int16"
    table = schema.to_arrow(out, "interaction_windows")
    assert table.schema.field("window_id").nullable is False
    with pytest.raises(KeyError):
        schema.conform(frame.drop(columns=["window_start_s"]), "interaction_windows")


def test_scores_and_confidences_declare_their_range() -> None:
    for f in schema.fields():
        if f.role in ("score", "confidence"):
            assert f.value_range == (0.0, 1.0), f.name
