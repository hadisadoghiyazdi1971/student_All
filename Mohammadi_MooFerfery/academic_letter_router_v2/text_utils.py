from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


PERSIAN_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
        "ـ": "",
    }
)

PERSIAN_STOPWORDS = {
    "از", "به", "در", "با", "برای", "که", "این", "آن", "را", "و", "یا", "یک", "می", "شود",
    "شده", "است", "هست", "بود", "باشد", "تا", "بر", "هم", "ما", "من", "شما", "ایشان", "خود",
    "خواهشمند", "لطفا", "لطفاً", "احترام", "سلام", "موضوع", "نامه", "درخواست", "بررسی", "مورد",
    "دانشگاه", "استاد", "جناب", "سرکار", "آقای", "خانم", "محترم", "با تشکر", "سپاس",
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "with", "on", "is", "are",
    "this", "that", "please", "dear", "regards", "university", "professor", "request", "letter",
}

GENERIC_KEYPHRASES = {
    "با سلام", "با احترام", "خواهشمند است", "لطفا بررسی", "لطفاً بررسی", "درخواست همکاری",
    "نامه دانشگاه", "استاد محترم", "جناب آقای", "سرکار خانم", "تشکر و احترام",
    "academic request", "dear professor", "best regards", "university request",
}


def normalize_persian_text(text: str, *, keep_newlines: bool = True) -> str:
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).translate(PERSIAN_TRANSLATION)
    value = value.replace("\u200c", " ").replace("\u200f", " ").replace("\u200e", " ")
    value = re.sub(r"[\t\r\f\v]+", " ", value)
    if keep_newlines:
        value = re.sub(r"[ ]{2,}", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        value = "\n".join(line.strip() for line in value.splitlines())
    else:
        value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_for_match(text: str) -> str:
    value = normalize_persian_text(text, keep_newlines=False).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def compact_text(text: str, limit: int = 12000) -> str:
    value = normalize_persian_text(text)
    if len(value) <= limit:
        return value
    head = value[: int(limit * 0.75)]
    tail = value[-int(limit * 0.25) :]
    return f"{head}\n\n[بخشی از متن برای رعایت محدودیت طول حذف شد]\n\n{tail}"


def detect_language(text: str) -> str:
    persian = len(re.findall(r"[\u0600-\u06FF]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    if persian and latin:
        ratio = min(persian, latin) / max(persian, latin)
        return "Mixed" if ratio >= 0.08 else ("Persian" if persian > latin else "English")
    if persian:
        return "Persian"
    if latin:
        return "English"
    return "Other"


def first_sentence(text: str, limit: int = 240) -> str:
    value = normalize_persian_text(text, keep_newlines=False)
    if not value:
        return ""
    parts = re.split(r"(?<=[.!؟?])\s+", value, maxsplit=1)
    sentence = parts[0].strip()
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


def deduplicate_preserve_order(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = str(item).strip()
        normalized = normalize_for_match(cleaned)
        if cleaned and normalized and normalized not in seen:
            seen.add(normalized)
            result.append(cleaned)
    return result


def tokenize_for_overlap(text: str) -> set[str]:
    normalized = normalize_for_match(text)
    return {
        token
        for token in normalized.split()
        if len(token) >= 2 and token not in PERSIAN_STOPWORDS
    }
