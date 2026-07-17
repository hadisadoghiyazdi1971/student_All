from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from models import FieldMatch, KeywordCandidate
from text_utils import deduplicate_preserve_order, normalize_for_match, normalize_persian_text


class TaxonomyDataError(RuntimeError):
    pass


@dataclass
class _EntryBuilder:
    canonical_name: str
    aliases: set[str] = field(default_factory=set)
    faculties: set[str] = field(default_factory=set)
    departments: set[str] = field(default_factory=set)
    specialties: set[str] = field(default_factory=set)
    contexts: set[str] = field(default_factory=set)
    people_count: int = 0


def _taxonomy_key(text: str) -> str:
    key = normalize_for_match(text).replace("ارتقاء", "ارتقا")
    replacements = {
        "نانو فناوری": "نانوفناوری",
        "ارتودانتیکس": "ارتودونتیکس",
    }
    for source, target in replacements.items():
        key = key.replace(source, target)
    return key


class TaxonomyService:
    def __init__(
        self,
        *,
        field_names_path: str | Path,
        medical_fields_path: str | Path,
        embedding_service,
    ) -> None:
        self.field_names_path = Path(field_names_path)
        self.medical_fields_path = Path(medical_fields_path)
        self.embedding_service = embedding_service
        self.entries = self._load_entries()
        self._contexts = [entry["context"] for entry in self.entries]
        fingerprint_source = "\n".join(self._contexts).encode("utf-8")
        self._fingerprint = hashlib.sha256(fingerprint_source).hexdigest()
        self._embeddings: np.ndarray | None = None

    @staticmethod
    def _safe(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return normalize_persian_text(str(value), keep_newlines=False)

    def _load_entries(self) -> list[dict]:
        if not self.field_names_path.exists():
            raise TaxonomyDataError(f"فایل حوزه‌ها پیدا نشد: {self.field_names_path}")
        if not self.medical_fields_path.exists():
            raise TaxonomyDataError(f"فایل جزئیات حوزه‌ها پیدا نشد: {self.medical_fields_path}")

        try:
            names_df = pd.read_excel(self.field_names_path)
            medical_df = pd.read_excel(self.medical_fields_path)
        except Exception as exc:  # noqa: BLE001
            raise TaxonomyDataError(f"خطا در خواندن فایل‌های taxonomy: {exc}") from exc

        builders: dict[str, _EntryBuilder] = {}

        def get_builder(display_name: str) -> _EntryBuilder | None:
            display = self._safe(display_name)
            key = _taxonomy_key(display)
            if not display or not key:
                return None
            if key not in builders:
                builders[key] = _EntryBuilder(canonical_name=display)
            return builders[key]

        name_column = "عنوان فیلد"
        if name_column not in names_df.columns:
            raise TaxonomyDataError(f"ستون «{name_column}» در فایل نام حوزه‌ها وجود ندارد.")

        for raw_name in names_df[name_column].tolist():
            builder = get_builder(raw_name)
            if builder is None:
                continue
            cleaned = self._safe(raw_name)
            builder.aliases.add(cleaned)
            builder.contexts.add(cleaned)

        required = {"دانشکده", "گروه آموزشی", "رشته / گرایش", "عنوان فیلد", "تعداد افراد"}
        missing = required - set(medical_df.columns)
        if missing:
            raise TaxonomyDataError(f"ستون‌های taxonomy ناقص هستند: {sorted(missing)}")

        for _, row in medical_df.iterrows():
            faculty = self._safe(row.get("دانشکده"))
            department = self._safe(row.get("گروه آموزشی"))
            specialty = self._safe(row.get("رشته / گرایش"))
            field_title = self._safe(row.get("عنوان فیلد"))
            canonical = specialty or field_title or department
            builder = get_builder(canonical)
            if builder is None:
                continue
            for alias in (canonical, field_title):
                if alias:
                    builder.aliases.add(alias)
            if faculty:
                builder.faculties.add(faculty)
            if department:
                builder.departments.add(department)
            if specialty:
                builder.specialties.add(specialty)
            try:
                builder.people_count += max(0, int(float(row.get("تعداد افراد", 0) or 0)))
            except (TypeError, ValueError):
                pass
            context_parts = [faculty, department, specialty, field_title]
            context = " | ".join(deduplicate_preserve_order(part for part in context_parts if part))
            if context:
                builder.contexts.add(context)

        entries: list[dict] = []
        for key, builder in builders.items():
            aliases = sorted(builder.aliases, key=lambda item: (len(item), item))
            context_parts = [
                f"حوزه: {builder.canonical_name}",
                f"نام های دیگر: {'، '.join(aliases)}" if aliases else "",
                f"دانشکده: {'، '.join(sorted(builder.faculties))}" if builder.faculties else "",
                f"گروه آموزشی: {'، '.join(sorted(builder.departments))}" if builder.departments else "",
                f"رشته و گرایش: {'، '.join(sorted(builder.specialties))}" if builder.specialties else "",
            ]
            context = " | ".join(part for part in context_parts if part)
            entries.append(
                {
                    "normalized_name": key,
                    "canonical_name": builder.canonical_name,
                    "aliases": aliases,
                    "faculties": sorted(builder.faculties),
                    "departments": sorted(builder.departments),
                    "specialties": sorted(builder.specialties),
                    "people_count": builder.people_count,
                    "context": context,
                }
            )
        entries.sort(key=lambda item: item["canonical_name"])
        return entries

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self._embeddings = self.embedding_service.get_or_create_corpus_embeddings(
                namespace="taxonomy",
                fingerprint=self._fingerprint,
                texts=self._contexts,
            )
        return self._embeddings

    def match_fields(
        self,
        *,
        letter_text: str,
        keyword_candidates: list[KeywordCandidate] | None = None,
        summary: str = "",
        top_k: int = 5,
        min_score: float = 0.35,
    ) -> list[FieldMatch]:
        if not self.entries:
            return []

        query_texts: list[str] = []
        weights: list[float] = []

        if letter_text.strip():
            query_texts.append(letter_text[:10000])
            weights.append(0.65 if not summary else 0.50)

        keyword_text = "، ".join(item.phrase for item in (keyword_candidates or []))
        if keyword_text:
            query_texts.append(keyword_text)
            weights.append(0.35 if not summary else 0.30)

        if summary.strip():
            query_texts.append(summary)
            weights.append(0.20)

        if not query_texts:
            return []

        weights_array = np.asarray(weights, dtype=np.float32)
        weights_array = weights_array / weights_array.sum()
        query_embeddings = self.embedding_service.encode_queries(query_texts)
        similarities = query_embeddings @ self.embeddings.T
        fused = np.average(similarities, axis=0, weights=weights_array)
        fused = np.clip(fused, 0.0, 1.0)

        ranked_indices = sorted(
            range(len(self.entries)),
            key=lambda index: (float(fused[index]), self.entries[index]["people_count"]),
            reverse=True,
        )

        matches: list[FieldMatch] = []
        for index in ranked_indices:
            score = float(fused[index])
            if score < min_score:
                continue
            entry = self.entries[index]
            evidence = [entry["canonical_name"]]
            evidence.extend(entry["aliases"][:3])
            matches.append(
                FieldMatch(
                    canonical_name=entry["canonical_name"],
                    score=score,
                    faculties=entry["faculties"],
                    departments=entry["departments"],
                    specialties=entry["specialties"][:5],
                    people_count=entry["people_count"],
                    evidence=deduplicate_preserve_order(evidence),
                )
            )
            if len(matches) >= top_k:
                break
        return matches
