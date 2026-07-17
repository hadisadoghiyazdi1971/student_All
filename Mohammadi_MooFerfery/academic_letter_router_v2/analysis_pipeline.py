from __future__ import annotations

import logging

from models import AnalysisResult, LetterAnalysis
from text_utils import deduplicate_preserve_order, detect_language, first_sentence, normalize_for_match, normalize_persian_text


logger = logging.getLogger(__name__)


class AnalysisPipelineError(RuntimeError):
    pass


class AnalysisPipeline:
    def __init__(self, *, keyword_service, taxonomy_service, ollama_service=None) -> None:
        self.keyword_service = keyword_service
        self.taxonomy_service = taxonomy_service
        self.ollama_service = ollama_service

    @staticmethod
    def _validate_selected_fields(analysis: LetterAnalysis, field_matches) -> bool:
        allowed = {normalize_for_match(item.canonical_name): item.canonical_name for item in field_matches}
        valid = []
        for selected in analysis.selected_fields:
            canonical = allowed.get(normalize_for_match(selected))
            if canonical and canonical not in valid:
                valid.append(canonical)
        academic_candidate = allowed.get(normalize_for_match(analysis.academic_field))
        if not valid and academic_candidate:
            valid.append(academic_candidate)
        analysis.selected_fields = valid
        if valid:
            analysis.academic_field = valid[0]
            return True
        analysis.academic_field = "نامشخص"
        analysis.confidence = min(analysis.confidence, 0.45)
        return False

    def analyze(
        self,
        letter_text: str,
        *,
        use_keybert: bool,
        use_ollama: bool,
        keyword_count: int,
        keyword_min_score: float,
        keyword_diversity: float,
        field_candidate_count: int,
        field_min_score: float,
    ) -> AnalysisResult:
        normalized_text = normalize_persian_text(letter_text)
        logger.info("Letter analysis started: chars=%s keybert=%s ollama=%s", len(normalized_text), use_keybert, use_ollama)
        if not normalized_text:
            raise AnalysisPipelineError("متن نامه خالی است.")
        if not use_keybert and not use_ollama:
            raise AnalysisPipelineError("حداقل یکی از گزینه‌های KeyBERT یا Ollama باید فعال باشد.")

        warnings: list[str] = []
        keyword_candidates = []
        if use_keybert:
            try:
                keyword_candidates = self.keyword_service.extract(
                    normalized_text,
                    max_keywords=keyword_count,
                    min_score=keyword_min_score,
                    diversity=keyword_diversity,
                )
            except Exception as exc:  # noqa: BLE001
                if not use_ollama:
                    raise AnalysisPipelineError(str(exc)) from exc
                warnings.append(f"KeyBERT اجرا نشد و تحلیل فقط با Ollama ادامه یافت: {exc}")

        initial_fields = self.taxonomy_service.match_fields(
            letter_text=normalized_text,
            keyword_candidates=keyword_candidates,
            top_k=field_candidate_count,
            min_score=field_min_score,
        )

        if use_ollama:
            if self.ollama_service is None:
                raise AnalysisPipelineError("سرویس Ollama پیکربندی نشده است.")
            try:
                analysis = self.ollama_service.analyze_letter(
                    normalized_text,
                    keyword_candidates,
                    initial_fields,
                )
            except Exception as exc:  # noqa: BLE001
                if not use_keybert:
                    raise AnalysisPipelineError(str(exc)) from exc
                warnings.append(f"Ollama اجرا نشد و نتیجه KeyBERT استفاده شد: {exc}")
                analysis = self._build_keybert_analysis(normalized_text, keyword_candidates, initial_fields)
            else:
                analysis.analysis_method = "combined" if keyword_candidates else "ollama"
                self._validate_selected_fields(analysis, initial_fields)
                analysis.keywords = deduplicate_preserve_order(
                    [*analysis.keywords, *(item.phrase for item in keyword_candidates)]
                )[:keyword_count]
                analysis.research_topics = deduplicate_preserve_order(analysis.research_topics)[:8]
                analysis.routing_tags = deduplicate_preserve_order(analysis.routing_tags)[:6]

            final_fields = self.taxonomy_service.match_fields(
                letter_text=normalized_text,
                keyword_candidates=keyword_candidates,
                summary=analysis.email_summary,
                top_k=field_candidate_count,
                min_score=field_min_score,
            )
            field_matches = final_fields or initial_fields
            validation_fields = list(field_matches)
            known = {normalize_for_match(item.canonical_name) for item in validation_fields}
            validation_fields.extend(
                item for item in initial_fields if normalize_for_match(item.canonical_name) not in known
            )
            self._validate_selected_fields(analysis, validation_fields)
        else:
            analysis = self._build_keybert_analysis(normalized_text, keyword_candidates, initial_fields)
            field_matches = initial_fields

        if not analysis.academic_field or analysis.academic_field in {"Unknown", "نامشخص"}:
            if (not use_ollama and field_matches) or (field_matches and field_matches[0].score >= max(0.50, field_min_score)):
                analysis.academic_field = field_matches[0].canonical_name
                if not analysis.selected_fields:
                    analysis.selected_fields = [field_matches[0].canonical_name]
            else:
                analysis.academic_field = "نامشخص"

        result = AnalysisResult(
            normalized_text=normalized_text,
            analysis=analysis,
            keyword_candidates=keyword_candidates,
            field_matches=field_matches,
            warnings=warnings,
        )
        logger.info("Letter analysis completed: keywords=%s fields=%s warnings=%s", len(keyword_candidates), len(field_matches), len(warnings))
        return result

    @staticmethod
    def _build_keybert_analysis(text, keyword_candidates, field_matches) -> LetterAnalysis:
        keywords = [item.phrase for item in keyword_candidates]
        selected_fields = [item.canonical_name for item in field_matches[:3]]
        confidence = field_matches[0].score * 0.8 if field_matches else (0.45 if keywords else 0.2)
        intent = "درخواست علمی یا دانشگاهی؛ نیازمند بررسی انسانی"
        lower = normalize_for_match(text)
        if any(term in lower for term in ("همکاری پژوهشی", "research collaboration", "پروژه پژوهشی")):
            intent = "همکاری پژوهشی"
        elif any(term in lower for term in ("پایان نامه", "پایان‌نامه", "thesis", "استاد راهنما")):
            intent = "راهنمایی پایان‌نامه"
        elif any(term in lower for term in ("اداری", "گواهی", "معاونت", "administrative")):
            intent = "درخواست اداری"

        return LetterAnalysis(
            language=detect_language(text),
            intent=intent,
            email_summary=first_sentence(text),
            academic_field=selected_fields[0] if selected_fields else "نامشخص",
            keywords=keywords,
            research_topics=keywords[:5],
            recommended_professor_profile=(
                f"عضو هیئت علمی فعال در حوزه {selected_fields[0]}" if selected_fields else "نیازمند بررسی کارشناس ارجاع"
            ),
            routing_tags=selected_fields[:3],
            selected_fields=selected_fields,
            confidence=max(0.0, min(1.0, confidence)),
            analysis_method="keybert",
        )
