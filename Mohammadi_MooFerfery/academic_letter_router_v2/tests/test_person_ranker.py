import pandas as pd

from person_ranker import rank_people


def test_rank_people_prefers_relevant_corresponding_author_and_explains():
    articles = pd.DataFrame(
        [
            {"id": 10, "title": "A", "relevance_score": 90.0, "citation": 10, "fwci": 1.5, "matched_keywords": ["قلب"]},
            {"id": 11, "title": "B", "relevance_score": 70.0, "citation": 2, "fwci": 0.8, "matched_keywords": ["عروق"]},
        ]
    )
    authors = pd.DataFrame(
        [
            {"article_id": 10, "person_id": 1, "is_first": 1, "is_corresponding": 1},
            {"article_id": 11, "person_id": 2, "is_first": 1, "is_corresponding": 0},
        ]
    )
    persons = pd.DataFrame([{"id": 1, "name": "الف"}, {"id": 2, "name": "ب"}])

    result = rank_people(articles, authors, persons)
    assert result.iloc[0]["person_name"] == "الف"
    assert 0 < result.iloc[0]["routing_score"] <= 100.0
    assert result.iloc[0]["routing_score"] > result.iloc[1]["routing_score"]
    assert any("نویسنده مسئول" in reason for reason in result.iloc[0]["explanations"])
