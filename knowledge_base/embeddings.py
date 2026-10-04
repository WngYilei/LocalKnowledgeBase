"""Embedding providers, including an offline local sentence-transformers backend."""

from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class HashEmbeddingProvider:
    """Deterministic fallback for tests and tiny offline demos, not production quality."""

    model_name = "hash-char-bigram-v1"
    dimension = 4096

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        characters = [char for char in text if not char.isspace()]
        tokens = characters + ["".join(characters[i : i + 2]) for i in range(len(characters) - 1)]
        for token in tokens:
            index = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big") % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class LocalEmbeddingProvider:
    """Run a downloaded sentence-transformers model without calling a remote API."""

    def __init__(self, model_path: str, batch_size: int = 16, device: str | None = None) -> None:
        if not model_path:
            raise ValueError("KNOWLEDGE_BASE_EMBEDDING_MODEL_PATH is required for local embeddings")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised in deployment, not unit tests
            raise RuntimeError(
                "Local embeddings require the optional dependency: pip install -e '.[local-embedding]'"
            ) from exc

        self.model_name = os.path.basename(os.path.normpath(model_path))
        self.batch_size = batch_size
        kwargs: dict[str, Any] = {"local_files_only": True}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(model_path, **kwargs)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    provider_name = os.getenv("KNOWLEDGE_BASE_EMBEDDING_PROVIDER", "hash").strip().lower()
    if provider_name == "hash":
        return HashEmbeddingProvider()
    if provider_name == "local":
        return LocalEmbeddingProvider(
            os.getenv("KNOWLEDGE_BASE_EMBEDDING_MODEL_PATH", ""),
            batch_size=int(os.getenv("KNOWLEDGE_BASE_EMBEDDING_BATCH_SIZE", "16")),
            device=os.getenv("KNOWLEDGE_BASE_EMBEDDING_DEVICE") or None,
        )
    raise ValueError(f"Unsupported embedding provider: {provider_name}")


def simple_embedding(text: str) -> list[float]:
    """Compatibility wrapper retained for existing callers and tests."""
    return HashEmbeddingProvider().embed_query(text)
