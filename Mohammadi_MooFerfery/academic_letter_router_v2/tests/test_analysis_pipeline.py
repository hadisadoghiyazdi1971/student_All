from analysis_pipeline import AnalysisPipeline
from models import FieldMatch, KeywordCandidate, LetterAnalysis


class FakeKeywordService:
    def extract(self, text, **kwargs):
        return [KeywordCandidate(phrase="تصویربرداری پزشکی", score=0.8)]


class FakeTaxonomyService:
    def match_fields(self, **kwargs):
        return [FieldMatch(canonical_name="رادیولوژی", score=0.75, departments=["رادیولوژی"])]


class FakeOllamaService:
    def analyze_letter(self, text, keywords, fields):
        return LetterAnalysis(
            language="Persian",
            intent="همکاری پژوهشی",
            email_summary="درخواست همکاری در تصویربرداری پزشکی",
            academic_field="رادیولوژی",
            keywords=["تصویربرداری پزشکی"],
            selected_fields=["رادیولوژی"],
            confidence=0.8,
        )


def test_combined_mode_uses_keybert_and_ollama():
    pipeline = AnalysisPipeline(
        keyword_service=FakeKeywordService(),
        taxonomy_service=FakeTaxonomyService(),
        ollama_service=FakeOllamaService(),
    )
    result = pipeline.analyze(
        "نامه درباره تصویربرداری پزشکی",
        use_keybert=True,
        use_ollama=True,
        keyword_count=8,
        keyword_min_score=0.25,
        keyword_diversity=0.45,
        field_candidate_count=5,
        field_min_score=0.35,
    )
    assert result.analysis.analysis_method == "combined"
    assert result.analysis.selected_fields == ["رادیولوژی"]
    assert result.keyword_candidates[0].phrase == "تصویربرداری پزشکی"
