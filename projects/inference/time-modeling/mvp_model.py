from __future__ import annotations

from pathlib import PurePath


def stable_model_identifier(
    model_id: str | None = None, model_path: str | None = None
) -> str:
    for candidate in (model_id, model_path):
        text = str(candidate or "").strip()
        if not text:
            continue
        normalized = text.rstrip("/\\")
        if not normalized:
            continue
        return PurePath(normalized).name or normalized
    return "unknown_model"
