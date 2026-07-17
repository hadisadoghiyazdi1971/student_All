from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from filelock import FileLock


logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    pass


class EmbeddingService:
    """Lazy SentenceTransformer wrapper with atomic on-disk corpus caching."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str | Path,
        device: str | None = None,
        use_disk_cache: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or os.getenv("EMBEDDING_DEVICE") or None
        if use_disk_cache is None:
            use_disk_cache = os.getenv("EMBEDDING_DISK_CACHE", "true").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.use_disk_cache = use_disk_cache
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingServiceError(
                    "کتابخانه sentence-transformers نصب نیست. requirements.txt را نصب کنید."
                ) from exc
            try:
                kwargs = {"device": self.device} if self.device else {}
                self._model = SentenceTransformer(self.model_name, **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingServiceError(
                    f"مدل embedding با نام {self.model_name} بارگذاری نشد: {exc}"
                ) from exc
        return self._model

    def _encode(self, texts: Sequence[str], *, query: bool, batch_size: int | None = None) -> np.ndarray:
        cleaned = [str(text or "").strip() for text in texts]
        if not cleaned:
            return np.empty((0, 0), dtype=np.float32)

        batch = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
        model = self.model
        try:
            method_name = "encode_query" if query else "encode_document"
            method = getattr(model, method_name, None)
            if callable(method):
                embeddings = method(
                    cleaned,
                    batch_size=batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            else:
                embeddings = model.encode(
                    cleaned,
                    batch_size=batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingServiceError(f"خطا در تولید embedding: {exc}") from exc
        return np.asarray(embeddings, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, query=True)

    def encode_documents(self, texts: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        return self._encode(texts, query=False, batch_size=batch_size)

    def cache_identity(self, namespace: str, fingerprint: str) -> str:
        raw = f"{namespace}|{self.model_name}|{fingerprint}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def get_or_create_corpus_embeddings(
        self,
        *,
        namespace: str,
        fingerprint: str,
        texts: Sequence[str],
    ) -> np.ndarray:
        if not self.use_disk_cache:
            logger.info(
                "Embedding disk cache disabled; encoding corpus directly: namespace=%s count=%s model=%s",
                namespace,
                len(texts),
                self.model_name,
            )
            return self.encode_documents(texts)

        identity = self.cache_identity(namespace, fingerprint)
        npz_path = self.cache_dir / f"{namespace}_{identity}.npz"
        metadata_path = self.cache_dir / f"{namespace}_{identity}.json"
        lock = FileLock(str(npz_path) + ".lock", timeout=600)

        with lock:
            if npz_path.exists() and metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    cached = np.load(npz_path, allow_pickle=False)["embeddings"]
                    if (
                        metadata.get("model_name") == self.model_name
                        and metadata.get("fingerprint") == fingerprint
                        and int(metadata.get("text_count", -1)) == len(texts)
                        and cached.shape[0] == len(texts)
                    ):
                        logger.info("Embedding cache hit: namespace=%s count=%s", namespace, len(texts))
                        return np.asarray(cached, dtype=np.float32)
                except Exception:
                    pass

            logger.info("Building embedding cache: namespace=%s count=%s model=%s", namespace, len(texts), self.model_name)
            embeddings = self.encode_documents(texts)
            tmp_npz = npz_path.with_suffix(".tmp.npz")
            tmp_json = metadata_path.with_suffix(".tmp.json")
            np.savez_compressed(tmp_npz, embeddings=embeddings)
            tmp_json.write_text(
                json.dumps(
                    {
                        "model_name": self.model_name,
                        "fingerprint": fingerprint,
                        "text_count": len(texts),
                        "dimensions": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            tmp_npz.replace(npz_path)
            tmp_json.replace(metadata_path)
            return embeddings
