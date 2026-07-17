from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class KeywordCandidate(BaseModel):
    phrase: str
    score: float = Field(ge=0.0, le=1.0)
    source: Literal["keybert", "ollama", "fallback"] = "keybert"


class FieldMatch(BaseModel):
    canonical_name: str
    score: float = Field(ge=0.0, le=1.0)
    faculties: list[str] = Field(default_factory=list)
    departments: list[str] = Field(default_factory=list)
    specialties: list[str] = Field(default_factory=list)
    people_count: int = Field(default=0, ge=0)
    evidence: list[str] = Field(default_factory=list)


class LetterAnalysis(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    language: str = "Unknown"
    intent: str = "Unknown"
    email_summary: str = ""
    academic_field: str = "Unknown"
    keywords: list[str] = Field(default_factory=list)
    research_topics: list[str] = Field(default_factory=list)
    recommended_professor_profile: str = ""
    routing_tags: list[str] = Field(default_factory=list)
    selected_fields: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    analysis_method: Literal["keybert", "ollama", "combined", "fallback"] = "fallback"

    @field_validator("keywords", "research_topics", "routing_tags", "selected_fields")
    @classmethod
    def clean_string_lists(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in value or []:
            cleaned = str(item).strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                result.append(cleaned)
                seen.add(key)
        return result


class Professor(BaseModel):
    """Compatibility model for the retired direct-profile matching path."""
    name: str
    title: str | None = None
    department: str | None = None
    email: str | None = None
    research_areas: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    raw_text: str | None = None


class ProfessorMatch(BaseModel):
    professor_name: str
    score: int = Field(ge=0, le=100)
    reason: str
    matched_keywords: list[str] = Field(default_factory=list)
    recommended_action: str = "route"


class RoutingResult(BaseModel):
    analysis: LetterAnalysis
    matches: list[ProfessorMatch] = Field(default_factory=list)
    final_recommendation: str = ""


class AnalysisResult(BaseModel):
    normalized_text: str
    analysis: LetterAnalysis
    keyword_candidates: list[KeywordCandidate] = Field(default_factory=list)
    field_matches: list[FieldMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LetterProposal(BaseModel):
    suggested_subject: str = ""
    recipient_title: str = ""
    suggested_letter: str = ""
    tone: str = "رسمی و علمی"
    improvement_notes: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    generation_method: Literal["ollama", "template"] = "template"


class AppSettings(BaseModel):
    top_articles: int = Field(default=30, ge=5, le=100)
    top_people: int = Field(default=5, ge=1, le=20)
    keyword_count: int = Field(default=8, ge=3, le=15)
    keyword_min_score: float = Field(default=0.25, ge=0.0, le=1.0)
    keyword_diversity: float = Field(default=0.45, ge=0.0, le=1.0)
    use_keybert: bool = True
    use_ollama: bool = True
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:8b"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    field_candidate_count: int = Field(default=5, ge=1, le=10)
    field_min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    proposal_tone: str = "رسمی و علمی"
    proposal_mode: Literal["rewrite", "new"] = "rewrite"
