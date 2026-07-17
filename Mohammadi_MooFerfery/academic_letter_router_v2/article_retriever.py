from __future__ import annotations

import hashlib
import logging
import os

import numpy as np
import pandas as pd

from models import FieldMatch, KeywordCandidate, LetterAnalysis
from text_utils import normalize_for_match


logger = logging.getLogger(__name__)


class ArticleRetrievalError(RuntimeError):
    pass


def _series_or_empty(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column].fillna("").astype(str)
    return pd.Series("", index=df.index, dtype="object")


def article_fingerprint(articles: pd.DataFrame) -> str:
    relevant = articles[[column for column in ("id", "search_text") if column in articles.columns]].copy()
    hashed = pd.util.hash_pandas_object(relevant, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def build_article_search_text(articles_df: pd.DataFrame) -> pd.DataFrame:
    articles = articles_df.copy().reset_index(drop=True)
    text_columns = [
        "title",
        "abstract",
        "journal_name",
        "authors",
        "keywords",
        "subject_areas",
        "academic_field",
    ]
    search_text = pd.Series("", index=articles.index, dtype="object")
    for column in text_columns:
        values = _series_or_empty(articles, column)
        search_text = search_text + " | " + values
    articles["search_text"] = search_text.str.strip(" |")
    return articles


def _keyword_overlap(search_text: str, keywords: list[str]) -> tuple[float, list[str]]:
    if not keywords:
        return 0.0, []
    normalized_text = normalize_for_match(search_text)
    matched: list[str] = []
    for keyword in keywords:
        normalized_keyword = normalize_for_match(keyword)
        if normalized_keyword and normalized_keyword in normalized_text:
            matched.append(keyword)
            continue
        keyword_tokens = set(normalized_keyword.split())
        text_tokens = set(normalized_text.split())
        if keyword_tokens and len(keyword_tokens & text_tokens) / len(keyword_tokens) >= 0.75:
            matched.append(keyword)
    return len(matched) / max(len(keywords), 1), matched


def retrieve_relevant_articles(
    *,
    letter_text: str,
    analysis: LetterAnalysis,
    articles_df: pd.DataFrame,
    embedding_service,
    keyword_candidates: list[KeywordCandidate] | None = None,
    field_matches: list[FieldMatch] | None = None,
    top_n: int = 30,
) -> pd.DataFrame:
    if articles_df is None or articles_df.empty:
        return pd.DataFrame()
    if "id" not in articles_df.columns:
        raise ArticleRetrievalError("ستون id در articles.csv وجود ندارد.")

    articles = build_article_search_text(articles_df)

    keywords = [item.phrase for item in (keyword_candidates or [])]
    keywords = list(dict.fromkeys([*analysis.keywords, *analysis.research_topics, *keywords]))
    candidate_fields = [item.canonical_name for item in (field_matches or [])]
    fields = list(dict.fromkeys([*analysis.selected_fields, *candidate_fields]))[:5]

    corpus_limit = max(0, int(os.getenv("ARTICLE_CORPUS_LIMIT", "0")))
    if corpus_limit and len(articles) > corpus_limit:
        # This is a smoke-test mode, not a production retrieval strategy. Prefer
        # rows with an exact keyword/field hit, then fill the remainder with a
        # deterministic sample so every run exercises the same corpus.
        terms = [term.strip() for term in [*keywords, *fields] if term.strip()]
        hit_mask = pd.Series(False, index=articles.index)
        for term in terms[:20]:
            hit_mask |= articles["search_text"].str.contains(term, case=False, regex=False, na=False)
        matched = articles.loc[hit_mask].head(corpus_limit)
        remaining = corpus_limit - len(matched)
        if remaining > 0:
            pool = articles.loc[~articles.index.isin(matched.index)]
            fill = pool.sample(n=min(remaining, len(pool)), random_state=42)
            articles = pd.concat([matched, fill], ignore_index=True)
        else:
            articles = matched.reset_index(drop=True)
        logger.warning(
            "Article corpus limited for fast test mode: using=%s total=%s",
            len(articles),
            len(articles_df),
        )

    query = "\n".join(
        part
        for part in (
            f"خلاصه: {analysis.email_summary}" if analysis.email_summary else "",
            f"حوزه‌ها: {'، '.join(fields or analysis.selected_fields)}" if (fields or analysis.selected_fields) else "",
            f"کلیدواژه‌ها: {'، '.join(keywords)}" if keywords else "",
            f"متن نامه: {letter_text[:8000]}",
        )
        if part
    )

    fingerprint = article_fingerprint(articles)
    corpus_embeddings = embedding_service.get_or_create_corpus_embeddings(
        namespace="articles",
        fingerprint=fingerprint,
        texts=articles["search_text"].tolist(),
    )
    query_embedding = embedding_service.encode_queries([query])[0]
    semantic_scores = np.clip(corpus_embeddings @ query_embedding, 0.0, 1.0)

    overlap_results = [_keyword_overlap(text, keywords) for text in articles["search_text"].tolist()]
    overlap_scores = np.asarray([item[0] for item in overlap_results], dtype=np.float32)
    articles["matched_keywords"] = [item[1] for item in overlap_results]

    semantic_weight = float(os.getenv("ARTICLE_SEMANTIC_WEIGHT", "0.88"))
    semantic_weight = min(1.0, max(0.0, semantic_weight))
    hybrid_scores = semantic_weight * semantic_scores + (1.0 - semantic_weight) * overlap_scores

    articles["semantic_score"] = semantic_scores * 100.0
    articles["keyword_overlap"] = overlap_scores * 100.0
    articles["relevance_score"] = hybrid_scores * 100.0

    result = (
        articles.sort_values(
            ["relevance_score", "semantic_score", "id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(int(top_n))
        .reset_index(drop=True)
    )
    logger.info("Article retrieval completed: corpus=%s returned=%s", len(articles), len(result))
    return result
