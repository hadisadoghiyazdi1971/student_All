from __future__ import annotations

import json
import logging
import os
from typing import TypeVar

import httpx
from pydantic import BaseModel

from json_utils import extract_json_object
from models import FieldMatch, KeywordCandidate, LetterAnalysis, LetterProposal
from prompts import (
    LETTER_ANALYSIS_SYSTEM_PROMPT,
    LETTER_ANALYSIS_USER_TEMPLATE,
    PROPOSAL_SYSTEM_PROMPT,
    PROPOSAL_USER_TEMPLATE,
)
from text_utils import compact_text


logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


class OllamaServiceError(RuntimeError):
    pass


class OllamaService:
    """Small, testable adapter for Ollama's native HTTP API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self.timeout_seconds = timeout_seconds or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

    def status(self) -> tuple[bool, str, list[str]]:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()
            models = [str(item.get("name", "")) for item in payload.get("models", []) if item.get("name")]
            if self.model not in models and not any(name.startswith(f"{self.model}:") for name in models):
                return False, f"Ollama فعال است ولی مدل {self.model} نصب نیست.", models
            return True, f"Ollama و مدل {self.model} آماده هستند.", models
        except Exception as exc:  # noqa: BLE001
            return False, f"اتصال به Ollama برقرار نشد: {exc}", []

    def _chat_structured(self, system_prompt: str, user_prompt: str, response_model: type[T], *, num_predict: int) -> T:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": response_model.model_json_schema(),
            "think": False,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
            "options": {
                "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
                "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
                "num_predict": int(num_predict),
            },
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                body = response.json()
            content = str(body.get("message", {}).get("content", ""))
            parsed = response_model.model_validate(extract_json_object(content))
            logger.info("Ollama structured response completed: model=%s schema=%s", self.model, response_model.__name__)
            return parsed
        except httpx.ConnectError as exc:
            raise OllamaServiceError(
                "Ollama در دسترس نیست. سرویس را اجرا کنید و OLLAMA_BASE_URL را بررسی کنید."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise OllamaServiceError(f"Ollama خطای HTTP برگرداند: {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OllamaServiceError(f"خروجی Ollama قابل پردازش نیست: {exc}") from exc

    def analyze_letter(
        self,
        letter_text: str,
        keyword_candidates: list[KeywordCandidate],
        field_matches: list[FieldMatch],
    ) -> LetterAnalysis:
        keybert_evidence = "\n".join(
            f"- {item.phrase}: {item.score:.3f}" for item in keyword_candidates
        ) or "- No KeyBERT evidence supplied"
        field_evidence = "\n".join(
            (
                f"- {item.canonical_name}: score={item.score:.3f}; "
                f"faculties={', '.join(item.faculties[:3]) or '-'}; "
                f"departments={', '.join(item.departments[:3]) or '-'}; "
                f"available_people={item.people_count}"
            )
            for item in field_matches
        ) or "- No taxonomy candidate passed the threshold"
        prompt = LETTER_ANALYSIS_USER_TEMPLATE.format(
            letter_text=compact_text(letter_text, int(os.getenv("LLM_ANALYSIS_MAX_CHARS", "10000"))),
            keybert_evidence=keybert_evidence,
            field_evidence=field_evidence,
        )
        return self._chat_structured(LETTER_ANALYSIS_SYSTEM_PROMPT, prompt, LetterAnalysis, num_predict=1000)

    def generate_proposal(
        self,
        *,
        letter_text: str,
        analysis: LetterAnalysis,
        field_matches: list[FieldMatch],
        people_evidence: str,
        tone: str,
        mode: str,
        recipient: str,
    ) -> LetterProposal:
        mode_label = "بازنویسی حرفه‌ای نامه موجود" if mode == "rewrite" else "پیش‌نویس جدید بر اساس محتوای نامه"
        prompt = PROPOSAL_USER_TEMPLATE.format(
            mode_label=mode_label,
            tone=tone,
            letter_text=compact_text(letter_text, int(os.getenv("LLM_PROPOSAL_MAX_CHARS", "9000"))),
            analysis_json=json.dumps(analysis.model_dump(), ensure_ascii=False, indent=2),
            fields_json=json.dumps([item.model_dump() for item in field_matches[:5]], ensure_ascii=False, indent=2),
            people_evidence=people_evidence or "هیچ فرد قطعی انتخاب نشده است.",
            recipient=recipient or "مدیریت محترم گروه آموزشی مرتبط",
        )
        proposal = self._chat_structured(PROPOSAL_SYSTEM_PROMPT, prompt, LetterProposal, num_predict=1800)
        proposal.generation_method = "ollama"
        return proposal
