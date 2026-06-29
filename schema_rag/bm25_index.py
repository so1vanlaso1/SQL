"""BM25 lexical index over normalized schema documents.

BM25 covers the literal/keyword half of hybrid retrieval - exact Vietnamese
không-dấu terms like ``khach hang``, ``doanh thu``, ``ngay dat hang``, ``ACTIVE`` -
which embeddings can blur. It indexes the same table/column/row documents as the
vector store, but over their *normalized* text so query and document share the
không-dấu space.

Persistence: a single pickle at ``config.BM25_INDEX_PATH`` holding the tokenized
corpus + ids/documents/metadata. The BM25Okapi model is rebuilt on load (cheap for
a few hundred docs) so the artifact is robust across rank_bm25 versions.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

from . import config
from .vn_text import tokenize
from .vectorstore import Hit

try:  # rank-bm25 is optional; retrieval degrades to vector-only without it.
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - exercised only when dependency is absent
    BM25Okapi = None


class BM25Index:
    def __init__(self, ids: List[str], documents: List[str], metadatas: List[dict], corpus_tokens: List[List[str]]):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self.corpus_tokens = corpus_tokens
        self._model = BM25Okapi(corpus_tokens) if (BM25Okapi and corpus_tokens) else None

    def search(self, query: str, k: int = 15) -> List[Hit]:
        if self._model is None or not self.ids:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._model.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: List[Hit] = []
        for i in ranked[: min(k, len(ranked))]:
            if scores[i] <= 0:
                continue
            hits.append(Hit(float(scores[i]), self.ids[i], self.documents[i], self.metadatas[i]))
        return hits

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or config.BM25_INDEX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "corpus_tokens": self.corpus_tokens,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        return path

    @classmethod
    def from_documents(cls, ids: List[str], documents: List[str], metadatas: List[dict]) -> "BM25Index":
        corpus_tokens = [tokenize(doc) for doc in documents]
        return cls(ids, documents, metadatas, corpus_tokens)


def build_bm25_index(ids: List[str], documents: List[str], metadatas: List[dict], path: Path | None = None) -> Optional[BM25Index]:
    """Build and persist a BM25 index. Returns None if rank-bm25 is unavailable."""
    if BM25Okapi is None:
        print("[bm25] rank-bm25 not installed; skipping BM25 index (retrieval stays vector-only).")
        return None
    index = BM25Index.from_documents(ids, documents, metadatas)
    out = index.save(path)
    print(f"[bm25] saved {len(ids)} lexical docs -> {out}")
    return index


def load_bm25_index(path: Path | None = None) -> Optional[BM25Index]:
    """Load the persisted BM25 index, or None if missing / unusable."""
    if BM25Okapi is None:
        return None
    p = Path(path or config.BM25_INDEX_PATH)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            payload = pickle.load(fh)
        return BM25Index(
            payload["ids"],
            payload["documents"],
            payload["metadatas"],
            payload["corpus_tokens"],
        )
    except Exception:
        return None
