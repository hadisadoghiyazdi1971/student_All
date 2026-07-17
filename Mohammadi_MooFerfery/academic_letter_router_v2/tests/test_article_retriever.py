import numpy as np
import pandas as pd

from article_retriever import retrieve_relevant_articles
from models import KeywordCandidate, LetterAnalysis


class FakeEmbeddingService:
    def get_or_create_corpus_embeddings(self, *, namespace, fingerprint, texts):
        # First article is aligned to the query, second is orthogonal.
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


def test_semantic_article_retrieval_orders_best_article_first():
    articles = pd.DataFrame(
        [
            {"id": 1, "title": "قلب و عروق", "abstract": "درمان بیماری قلبی"},
            {"id": 2, "title": "زبان شناسی", "abstract": "تحلیل متن"},
        ]
    )
    analysis = LetterAnalysis(email_summary="درخواست درباره قلب", keywords=["قلب"])
    result = retrieve_relevant_articles(
        letter_text="درخواست بررسی بیماری قلبی",
        analysis=analysis,
        articles_df=articles,
        embedding_service=FakeEmbeddingService(),
        keyword_candidates=[KeywordCandidate(phrase="قلب", score=0.8)],
        top_n=2,
    )
    assert result.iloc[0]["id"] == 1
    assert result.iloc[0]["relevance_score"] > result.iloc[1]["relevance_score"]
