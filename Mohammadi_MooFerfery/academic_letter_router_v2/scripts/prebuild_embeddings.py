from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PATHS
from article_retriever import article_fingerprint, build_article_search_text
from data_loader import load_university_data
from embedding_service import EmbeddingService
from taxonomy_service import TaxonomyService


def main() -> int:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    service = EmbeddingService(model_name, PATHS.cache_dir)
    taxonomy = TaxonomyService(
        field_names_path=PATHS.taxonomy_dir / "university_37_field_names_only.xlsx",
        medical_fields_path=PATHS.taxonomy_dir / "university_37_medical_fields_only.xlsx",
        embedding_service=service,
    )
    print(f"Building taxonomy index for {len(taxonomy.entries)} canonical fields...")
    _ = taxonomy.embeddings

    articles = build_article_search_text(load_university_data(PATHS.data_dir)["articles"])
    fingerprint = article_fingerprint(articles)
    print(f"Building article index for {len(articles)} articles...")
    service.get_or_create_corpus_embeddings(
        namespace="articles",
        fingerprint=fingerprint,
        texts=articles["search_text"].tolist(),
    )
    print("Embedding indexes are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
