from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_profile_records(db_path: str | Path) -> list[dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        records.append(json.loads(text))
    return records


def append_profile_record(db_path: str | Path, record: dict[str, Any]) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def match_profile_record(
    records: list[dict[str, Any]],
    component: str,
    match_fields: dict[str, Any],
) -> dict[str, Any] | None:
    matched: dict[str, Any] | None = None
    for record in records:
        if record.get("component") != component:
            continue
        if any(record.get(key) != value for key, value in match_fields.items()):
            continue
        matched = record
    return matched
