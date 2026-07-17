from __future__ import annotations

import os

from config import PATHS
from data_loader import load_university_data
from document_reader import get_ocr_status
from embedding_service import EmbeddingService
from llm_service import OllamaService
from taxonomy_service import TaxonomyService


def main() -> int:
    exit_code = 0
    print(f"Data directory: {PATHS.data_dir}")
    print(f"Taxonomy directory: {PATHS.taxonomy_dir}")
    try:
        data = load_university_data(PATHS.data_dir)
        print(f"Data OK: persons={len(data['persons'])}, articles={len(data['articles'])}, relations={len(data['article_authors'])}")
    except Exception as exc:  # noqa: BLE001
        print(f"Data ERROR: {exc}")
        exit_code = 1

    try:
        taxonomy = TaxonomyService(
            field_names_path=PATHS.taxonomy_dir / "university_37_field_names_only.xlsx",
            medical_fields_path=PATHS.taxonomy_dir / "university_37_medical_fields_only.xlsx",
            embedding_service=EmbeddingService(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"), PATHS.cache_dir),
        )
        print(f"Taxonomy OK: canonical_fields={len(taxonomy.entries)}")
    except Exception as exc:  # noqa: BLE001
        print(f"Taxonomy ERROR: {exc}")
        exit_code = 1

    ocr_ok, ocr_message = get_ocr_status()
    print(f"OCR {'OK' if ocr_ok else 'WARNING'}: {ocr_message}")

    ollama_ok, message, models = OllamaService().status()
    print(f"Ollama {'OK' if ollama_ok else 'ERROR'}: {message}")
    if models:
        print("Installed Ollama models:", ", ".join(models))
    if not ollama_ok:
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
