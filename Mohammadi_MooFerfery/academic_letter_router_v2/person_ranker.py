from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "person_id",
    "person_name",
    "total_score",
    "routing_score",
    "related_articles_count",
    "avg_relevance",
    "first_author_count",
    "corresponding_author_count",
    "top_articles",
    "matched_keywords",
    "explanations",
]


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, (tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [str(value)] if str(value).strip() else []


def rank_people(
    relevant_articles: pd.DataFrame,
    article_authors: pd.DataFrame,
    persons: pd.DataFrame,
    *,
    max_articles_per_person: int = 10,
) -> pd.DataFrame:
    if relevant_articles is None or relevant_articles.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    required_articles = {"id", "relevance_score"}
    required_authors = {"article_id", "person_id"}
    required_persons = {"id", "name"}
    missing = {
        "articles": required_articles - set(relevant_articles.columns),
        "article_authors": required_authors - set(article_authors.columns),
        "persons": required_persons - set(persons.columns),
    }
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise ValueError(f"ستون‌های لازم برای رتبه‌بندی ناقص هستند: {missing}")

    article_columns = [
        column
        for column in (
            "id", "title", "relevance_score", "semantic_score", "citation", "fwci", "matched_keywords"
        )
        if column in relevant_articles.columns
    ]
    author_columns = [
        column
        for column in ("article_id", "person_id", "is_first", "is_corresponding")
        if column in article_authors.columns
    ]

    merged = (
        relevant_articles[article_columns]
        .rename(columns={"id": "article_id"})
        .merge(article_authors[author_columns], on="article_id", how="inner", validate="one_to_many")
        .merge(
            persons[["id", "name"]].rename(columns={"id": "person_id", "name": "person_name"}),
            on="person_id",
            how="inner",
            validate="many_to_one",
        )
    )
    if merged.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for column in ("citation", "fwci", "is_first", "is_corresponding", "relevance_score"):
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    merged["citation"] = merged["citation"].clip(lower=0, upper=100000)
    merged["fwci"] = merged["fwci"].clip(lower=0, upper=10)
    merged["author_weight"] = 1.0 + 0.20 * (merged["is_first"] == 1) + 0.30 * (merged["is_corresponding"] == 1)
    merged["quality_weight"] = (
        1.0
        + np.log1p(merged["citation"]) * 0.05
        + merged["fwci"] * 0.05
    ).clip(upper=2.5)
    merged["final_article_score"] = (
        merged["relevance_score"].clip(lower=0)
        * merged["author_weight"]
        * merged["quality_weight"]
    )

    rows: list[dict[str, Any]] = []
    for person_id, group in merged.groupby("person_id", sort=False):
        ranked = group.sort_values("final_article_score", ascending=False).head(max_articles_per_person).copy()
        ranked["evidence_rank"] = np.arange(1, len(ranked) + 1)
        ranked["discounted_score"] = ranked["final_article_score"] / np.sqrt(ranked["evidence_rank"])

        total_score = float(ranked["discounted_score"].sum())
        related_count = int(group["article_id"].nunique())
        avg_relevance = float(group["relevance_score"].mean())
        first_count = int((group["is_first"] == 1).sum())
        corresponding_count = int((group["is_corresponding"] == 1).sum())
        top_articles = [str(item) for item in ranked.get("title", pd.Series(dtype=str)).fillna("").head(3) if str(item).strip()]

        keyword_scores: dict[str, float] = {}
        if "matched_keywords" in ranked.columns:
            for _, article in ranked.iterrows():
                for keyword in _as_list(article.get("matched_keywords")):
                    keyword_scores[keyword] = max(keyword_scores.get(keyword, 0.0), float(article["relevance_score"]))
        matched_keywords = [item[0] for item in sorted(keyword_scores.items(), key=lambda pair: pair[1], reverse=True)[:6]]

        evidence_weights = 1.0 / np.sqrt(ranked["evidence_rank"].to_numpy(dtype=float))
        weighted_relevance = float(
            np.average(ranked["relevance_score"].to_numpy(dtype=float), weights=evidence_weights)
        )
        role_bonus = min(8.0, corresponding_count * 2.0 + first_count * 1.0)
        quality_bonus = min(8.0, max(0.0, float(ranked["quality_weight"].mean()) - 1.0) * 12.0)
        breadth_bonus = min(6.0, math.log1p(related_count) * 2.0)
        routing_score = min(
            100.0,
            max(0.0, 0.85 * weighted_relevance + role_bonus + quality_bonus + breadth_bonus),
        )

        explanations = [
            f"{related_count} مقاله مرتبط در سوابق این فرد یافت شد.",
            f"میانگین ارتباط مقاله‌ها با نامه {avg_relevance:.1f} از 100 است.",
        ]
        if corresponding_count:
            explanations.append(f"در {corresponding_count} مقاله مرتبط نویسنده مسئول بوده است.")
        if first_count:
            explanations.append(f"در {first_count} مقاله مرتبط نویسنده اول بوده است.")
        if matched_keywords:
            explanations.append("کلیدواژه‌های مشترک: " + "، ".join(matched_keywords[:4]))

        rows.append(
            {
                "person_id": person_id,
                "person_name": str(group["person_name"].iloc[0]),
                "total_score": total_score,
                "routing_score": routing_score,
                "related_articles_count": related_count,
                "avg_relevance": avg_relevance,
                "first_author_count": first_count,
                "corresponding_author_count": corresponding_count,
                "top_articles": top_articles,
                "matched_keywords": matched_keywords,
                "explanations": explanations,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["total_score", "avg_relevance", "related_articles_count", "person_name"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return result[OUTPUT_COLUMNS]
