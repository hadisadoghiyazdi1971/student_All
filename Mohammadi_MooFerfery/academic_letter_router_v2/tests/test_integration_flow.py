from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis_pipeline import AnalysisPipeline
from article_retriever import retrieve_relevant_articles
from models import KeywordCandidate, LetterAnalysis
from person_ranker import rank_people
from proposal_service import ProposalService
from taxonomy_service import TaxonomyService


class HashEmbeddingService:
    dimensions = 24

    @classmethod
    def _encode(cls, texts):
        rows = []
        for text in texts:
            vector = np.zeros(cls.dimensions, dtype=np.float32)
            for token in str(text).split():
                vector[hash(token) % cls.dimensions] += 1.0
            norm = np.linalg.norm(vector)
            rows.append(vector / norm if norm else vector)
        return np.asarray(rows, dtype=np.float32)

    def encode_queries(self, texts):
        return self._encode(texts)

    def encode_documents(self, texts, **kwargs):
        return self._encode(texts)

    def get_or_create_corpus_embeddings(self, *, namespace, fingerprint, texts):
        return self._encode(texts)


class FakeKeywordService:
    def extract(self, text, **kwargs):
        return [
            KeywordCandidate(phrase="قلب و عروق", score=0.85),
            KeywordCandidate(phrase="بیماری قلبی", score=0.78),
        ]


class FakeOllamaService:
    def analyze_letter(self, text, keywords, fields):
        selected = fields[0].canonical_name if fields else "نامشخص"
        return LetterAnalysis(
            language="Persian",
            intent="همکاری پژوهشی",
            email_summary="درخواست همکاری پژوهشی درباره بیماری‌های قلبی",
            academic_field=selected,
            keywords=["قلب و عروق", "بیماری قلبی"],
            research_topics=["تشخیص بیماری قلبی"],
            selected_fields=[selected] if fields else [],
            confidence=0.82,
        )


def test_full_flow_with_real_taxonomy_files():
    root = Path(__file__).resolve().parents[1]
    embedding = HashEmbeddingService()
    taxonomy = TaxonomyService(
        field_names_path=root / "data/taxonomy/university_37_field_names_only.xlsx",
        medical_fields_path=root / "data/taxonomy/university_37_medical_fields_only.xlsx",
        embedding_service=embedding,
    )
    pipeline = AnalysisPipeline(
        keyword_service=FakeKeywordService(),
        taxonomy_service=taxonomy,
        ollama_service=FakeOllamaService(),
    )
    analysis_result = pipeline.analyze(
        "برای یک پژوهش درباره تشخیص بیماری قلبی درخواست همکاری دارم.",
        use_keybert=True,
        use_ollama=True,
        keyword_count=8,
        keyword_min_score=0.25,
        keyword_diversity=0.45,
        field_candidate_count=5,
        field_min_score=0.0,
    )

    articles = pd.DataFrame(
        [
            {"id": 1, "title": "تشخیص بیماری قلبی", "abstract": "قلب و عروق", "citation": 20, "fwci": 1.4},
            {"id": 2, "title": "موضوع نامرتبط", "abstract": "زبان شناسی", "citation": 1, "fwci": 0.3},
        ]
    )
    relevant = retrieve_relevant_articles(
        letter_text=analysis_result.normalized_text,
        analysis=analysis_result.analysis,
        articles_df=articles,
        embedding_service=embedding,
        keyword_candidates=analysis_result.keyword_candidates,
        field_matches=analysis_result.field_matches,
        top_n=2,
    )
    authors = pd.DataFrame(
        [
            {"article_id": 1, "person_id": 10, "is_first": 1, "is_corresponding": 1},
            {"article_id": 2, "person_id": 11, "is_first": 1, "is_corresponding": 0},
        ]
    )
    persons = pd.DataFrame([{"id": 10, "name": "دکتر الف"}, {"id": 11, "name": "دکتر ب"}])
    ranked = rank_people(relevant, authors, persons)
    proposal = ProposalService().generate(
        original_text=analysis_result.normalized_text,
        result=analysis_result,
        ranked_people=ranked,
        use_ollama=False,
        tone="رسمی و علمی",
        mode="new",
        audience="field",
    )

    assert analysis_result.keyword_candidates
    assert analysis_result.field_matches
    assert not relevant.empty
    assert not ranked.empty
    assert proposal.suggested_letter
    assert proposal.generation_method == "template"


def test_proposal_docx_is_generated():
    from models import LetterProposal
    from proposal_service import proposal_to_docx_bytes

    content = proposal_to_docx_bytes(
        LetterProposal(
            suggested_subject="موضوع آزمایشی",
            recipient_title="مدیریت محترم گروه",
            suggested_letter="با سلام\nمتن فارسی آزمایشی",
        )
    )
    assert content.startswith(b"PK")
    assert len(content) > 1000
