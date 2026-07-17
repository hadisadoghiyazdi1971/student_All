from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd


logger = logging.getLogger(__name__)


class UniversityDataError(RuntimeError):
    pass


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"فایل پیدا نشد: {path}")
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1256"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(str(exc))
    raise UniversityDataError(f"رمزگذاری فایل {path.name} قابل تشخیص نیست: {errors[-1] if errors else ''}")


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result


def _rename_first_available(df: pd.DataFrame, target: str, aliases: list[str]) -> pd.DataFrame:
    if target in df.columns:
        return df
    normalized = {str(column).strip().casefold(): column for column in df.columns}
    for alias in aliases:
        source = normalized.get(alias.casefold())
        if source is not None:
            return df.rename(columns={source: target})
    return df


def _normalize_identifier(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text or None


def _normalize_identifier_column(df: pd.DataFrame, column: str) -> None:
    df[column] = df[column].map(_normalize_identifier).astype("string")


def _require_columns(df: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise UniversityDataError(f"ستون‌های ضروری در {filename} وجود ندارند: {sorted(missing)}")


def load_university_data(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    base = Path(data_dir)
    persons = _strip_columns(_read_csv(base / "persons.csv"))
    articles = _strip_columns(_read_csv(base / "articles.csv"))
    article_authors = _strip_columns(_read_csv(base / "article_authors.csv"))

    persons = _rename_first_available(persons, "id", ["person_id", "Person_ID", "شناسه فرد"])
    persons = _rename_first_available(persons, "name", ["person_name", "full_name", "نام", "نام و نام خانوادگی"])
    articles = _rename_first_available(articles, "id", ["article_id", "Article_ID", "شناسه مقاله"])
    articles = _rename_first_available(articles, "title", ["article_title", "عنوان", "عنوان مقاله"])
    article_authors = _rename_first_available(article_authors, "article_id", ["Article_ID", "article", "شناسه مقاله"])
    article_authors = _rename_first_available(article_authors, "person_id", ["Person_ID", "person", "شناسه فرد"])
    article_authors = _rename_first_available(article_authors, "is_first", ["is_first_author", "first_author"])
    article_authors = _rename_first_available(article_authors, "is_corresponding", ["is_corresponding_author", "corresponding_author"])

    _require_columns(persons, {"id", "name"}, "persons.csv")
    _require_columns(articles, {"id", "title"}, "articles.csv")
    _require_columns(article_authors, {"article_id", "person_id"}, "article_authors.csv")

    _normalize_identifier_column(persons, "id")
    _normalize_identifier_column(articles, "id")
    _normalize_identifier_column(article_authors, "article_id")
    _normalize_identifier_column(article_authors, "person_id")

    if persons["id"].isna().any() or persons["id"].duplicated().any():
        raise UniversityDataError("ستون id در persons.csv باید غیرخالی و یکتا باشد.")
    if articles["id"].isna().any() or articles["id"].duplicated().any():
        raise UniversityDataError("ستون id در articles.csv باید غیرخالی و یکتا باشد.")
    persons["name"] = persons["name"].fillna("").astype(str).str.strip()
    if (persons["name"] == "").any():
        raise UniversityDataError("نام فرد در persons.csv نباید خالی باشد.")

    for column in ("citation", "fwci"):
        if column not in articles.columns:
            articles[column] = 0.0
        articles[column] = pd.to_numeric(articles[column], errors="coerce").fillna(0.0)

    for column in ("is_first", "is_corresponding"):
        if column not in article_authors.columns:
            article_authors[column] = 0
        article_authors[column] = pd.to_numeric(article_authors[column], errors="coerce").fillna(0).astype(int)

    article_authors = article_authors.dropna(subset=["article_id", "person_id"]).drop_duplicates().reset_index(drop=True)
    valid_article_ids = set(articles["id"].dropna().astype(str))
    valid_person_ids = set(persons["id"].dropna().astype(str))
    invalid_article_links = ~article_authors["article_id"].astype(str).isin(valid_article_ids)
    invalid_person_links = ~article_authors["person_id"].astype(str).isin(valid_person_ids)
    invalid_links = invalid_article_links | invalid_person_links
    if invalid_links.any():
        logger.warning("Dropping invalid article-author links: count=%s", int(invalid_links.sum()))
        article_authors = article_authors.loc[~invalid_links].reset_index(drop=True)

    optional_files = {
        "groups": "groups.csv",
        "faculties": "faculties.csv",
        "group_faculty": "group_faculty.csv",
        "education_majors": "education_majors.csv",
        "academic_ranks": "academic_ranks.csv",
        "metrics": "metrics.csv",
    }
    data = {
        "persons": persons.reset_index(drop=True),
        "articles": articles.reset_index(drop=True),
        "article_authors": article_authors,
    }
    for key, filename in optional_files.items():
        path = base / filename
        data[key] = _strip_columns(_read_csv(path)) if path.exists() else pd.DataFrame()
    logger.info("University data loaded: persons=%s articles=%s relations=%s", len(persons), len(articles), len(article_authors))
    return data
