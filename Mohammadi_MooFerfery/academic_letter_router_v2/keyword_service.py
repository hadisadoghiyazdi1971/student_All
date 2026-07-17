from __future__ import annotations

from collections import Counter

from models import KeywordCandidate
from text_utils import (
    GENERIC_KEYPHRASES,
    PERSIAN_STOPWORDS,
    normalize_for_match,
    normalize_persian_text,
)


class KeywordServiceError(RuntimeError):
    pass


class KeywordService:
    def __init__(self, embedding_service) -> None:
        self.embedding_service = embedding_service
        self._keybert = None

    @property
    def model(self):
        if self._keybert is None:
            try:
                from keybert import KeyBERT
            except ImportError as exc:
                raise KeywordServiceError(
                    "کتابخانه keybert نصب نیست. requirements.txt را نصب کنید."
                ) from exc
            try:
                self._keybert = KeyBERT(model=self.embedding_service.model)
            except Exception as exc:  # noqa: BLE001
                raise KeywordServiceError(f"KeyBERT راه‌اندازی نشد: {exc}") from exc
        return self._keybert

    @staticmethod
    def _is_redundant(candidate: str, selected: list[str]) -> bool:
        candidate_tokens = set(normalize_for_match(candidate).split())
        if not candidate_tokens:
            return True
        for existing in selected:
            existing_tokens = set(normalize_for_match(existing).split())
            union = candidate_tokens | existing_tokens
            overlap = len(candidate_tokens & existing_tokens) / max(len(union), 1)
            if overlap >= 0.78:
                return True
            candidate_norm = " ".join(candidate_tokens)
            existing_norm = " ".join(existing_tokens)
            if candidate_norm == existing_norm:
                return True
        return False

    def extract(
        self,
        text: str,
        *,
        max_keywords: int = 8,
        min_score: float = 0.25,
        diversity: float = 0.45,
    ) -> list[KeywordCandidate]:
        normalized = normalize_persian_text(text, keep_newlines=False)
        if len(normalized) < 20:
            return self.frequency_fallback(normalized, max_keywords=max_keywords)

        try:
            from sklearn.feature_extraction.text import CountVectorizer

            vectorizer = CountVectorizer(
                ngram_range=(1, 3),
                stop_words=sorted(word for word in PERSIAN_STOPWORDS if " " not in word),
                lowercase=True,
                token_pattern=r"(?u)\b[\w‌-]{2,}\b",
                max_features=5000,
            )
            candidate_count = min(45, max(max_keywords * 3, max_keywords + 8))
            raw_keywords = self.model.extract_keywords(
                normalized,
                vectorizer=vectorizer,
                top_n=candidate_count,
                use_mmr=True,
                diversity=diversity,
            )
        except ValueError:
            return self.frequency_fallback(normalized, max_keywords=max_keywords)
        except Exception as exc:  # noqa: BLE001
            raise KeywordServiceError(f"استخراج کلمات کلیدی ناموفق بود: {exc}") from exc

        selected: list[str] = []
        results: list[KeywordCandidate] = []
        generic_normalized = {normalize_for_match(item) for item in GENERIC_KEYPHRASES}

        for phrase, raw_score in sorted(raw_keywords, key=lambda item: float(item[1]), reverse=True):
            cleaned = normalize_persian_text(str(phrase), keep_newlines=False).strip(" -،,.؛:")
            normalized_phrase = normalize_for_match(cleaned)
            score = max(0.0, min(1.0, float(raw_score)))
            if not cleaned or score < min_score:
                continue
            if normalized_phrase in generic_normalized:
                continue
            tokens = [token for token in normalized_phrase.split() if token not in PERSIAN_STOPWORDS]
            if not tokens or len(normalized_phrase) < 3:
                continue
            if self._is_redundant(cleaned, selected):
                continue
            selected.append(cleaned)
            results.append(KeywordCandidate(phrase=cleaned, score=score, source="keybert"))
            if len(results) >= max_keywords:
                break

        return results or self.frequency_fallback(normalized, max_keywords=max_keywords)

    @staticmethod
    def frequency_fallback(text: str, *, max_keywords: int) -> list[KeywordCandidate]:
        normalized = normalize_for_match(text)
        tokens = [
            token
            for token in normalized.split()
            if len(token) >= 3 and token not in PERSIAN_STOPWORDS and not token.isdigit()
        ]
        counts = Counter(tokens)
        if not counts:
            return []
        max_count = max(counts.values())
        return [
            KeywordCandidate(
                phrase=token,
                score=min(1.0, 0.25 + 0.5 * count / max_count),
                source="fallback",
            )
            for token, count in counts.most_common(max_keywords)
        ]
