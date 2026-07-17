from pathlib import Path

from data_loader import load_university_data


def test_data_loader_normalizes_identifier_types_and_drops_invalid_links(tmp_path: Path):
    (tmp_path / "persons.csv").write_text("id,name\n1,الف\n2,ب\n", encoding="utf-8")
    (tmp_path / "articles.csv").write_text("id,title\n10,مقاله\n", encoding="utf-8")
    (tmp_path / "article_authors.csv").write_text(
        "article_id,person_id,is_first,is_corresponding\n10.0,1.0,1,1\n99,2,0,0\n",
        encoding="utf-8",
    )
    data = load_university_data(tmp_path)
    assert data["persons"]["id"].tolist() == ["1", "2"]
    assert len(data["article_authors"]) == 1
    assert data["article_authors"].iloc[0]["article_id"] == "10"
