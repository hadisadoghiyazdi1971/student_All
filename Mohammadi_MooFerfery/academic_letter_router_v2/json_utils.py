from __future__ import annotations

import json
import re
from typing import Any


class JSONParseError(ValueError):
    pass


def extract_json_object(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise JSONParseError("پاسخ مدل خالی است.")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise JSONParseError("هیچ شیء JSON در پاسخ مدل پیدا نشد.")
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise JSONParseError(f"JSON تولیدشده توسط مدل معتبر نیست: {exc}") from exc

    if not isinstance(data, dict):
        raise JSONParseError("خروجی مورد انتظار باید یک شیء JSON باشد.")
    return data
