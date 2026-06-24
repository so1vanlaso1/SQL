"""Embedding model wrapper.

Primary  : ibm-granite/granite-embedding-311m-multilingual-r2 via sentence-transformers.
Fallback : a deterministic, dependency-free hashing embedder so the *pipeline* always
           runs end-to-end even on machines without torch (e.g. brand-new Python where
           wheels are missing). The fallback is lower quality - use the real model for
           anything beyond a smoke test.

All vectors are L2-normalized so cosine similarity == dot product.
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

from . import config


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SentenceTransformerEmbedder:
    """Real embedding model (Granite by default)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = int(self.model.get_embedding_dimension())
        else:
            self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder:
    """Dependency-free fallback: hashed character n-grams -> fixed-dim vector.

    Not semantic like a transformer, but good enough to demonstrate the retrieval
    plumbing offline. Deterministic across runs.
    """

    def __init__(self, dim: int = 768, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram
        self.model_name = f"hashing-{dim}d"

    def _vec(self, text: str) -> np.ndarray:
        text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        tokens = text.split()
        v = np.zeros(self.dim, dtype=np.float32)
        grams: List[str] = list(tokens)  # whole words
        for tok in tokens:                # + char n-grams for sub-word matching
            padded = f"#{tok}#"
            for i in range(len(padded) - self.ngram + 1):
                grams.append(padded[i : i + self.ngram])
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            v[idx] += sign
        return v

    def encode(self, texts: List[str]) -> np.ndarray:
        return _normalize(np.vstack([self._vec(t) for t in texts]))


_INSTANCE = None


def get_embedder():
    """Return a singleton embedder according to config.EMBEDDER."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    mode = config.EMBEDDER
    if mode in ("st", "auto"):
        try:
            _INSTANCE = SentenceTransformerEmbedder(config.EMBED_MODEL)
            print(f"[embedder] using sentence-transformers: {config.EMBED_MODEL} (dim={_INSTANCE.dim})")
            return _INSTANCE
        except Exception as exc:  # noqa: BLE001
            if mode == "st":
                raise
            print(
                f"[embedder] sentence-transformers/{config.EMBED_MODEL} unavailable "
                f"({exc.__class__.__name__}: {exc}). Falling back to hashing embedder.\n"
                f"           -> install deps (see requirements.txt) and use Python 3.12 "
                f"to enable the real Granite model."
            )
    _INSTANCE = HashingEmbedder()
    print(f"[embedder] using fallback: {_INSTANCE.model_name}")
    return _INSTANCE
