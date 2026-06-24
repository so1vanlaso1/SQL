"""A tiny, local, persistent vector store (cosine similarity over normalized vectors).

Why not Chroma/Qdrant/pgvector here? For a 20-table schema the whole "vector DB" is a
few hundred rows - a numpy matrix is faster, has zero external deps, and is trivial to
read. The interface (add / search / save / load) mirrors a real vector DB so you can
swap in Chroma/Qdrant/pgvector later without touching the rest of the pipeline.

Persistence: <INDEX_DIR>/vectors.npy + <INDEX_DIR>/meta.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Hit:
    score: float
    doc_id: str
    document: str
    metadata: dict


@dataclass
class VectorStore:
    dim: int
    model_name: str = "unknown"
    _vectors: Optional[np.ndarray] = None
    ids: List[str] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)
    metadatas: List[dict] = field(default_factory=list)

    # ---- write ----
    def add(self, ids: List[str], vectors: np.ndarray, documents: List[str], metadatas: List[dict]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    # ---- read ----
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Hit]:
        if self._vectors is None or len(self.ids) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        sims = self._vectors @ q  # vectors are normalized -> dot == cosine
        k = min(k, len(self.ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [
            Hit(float(sims[i]), self.ids[i], self.documents[i], self.metadatas[i])
            for i in top
        ]

    # ---- persistence ----
    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self._vectors if self._vectors is not None else np.zeros((0, self.dim), np.float32))
        meta = {
            "dim": self.dim,
            "model_name": self.model_name,
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        store = cls(dim=meta["dim"], model_name=meta.get("model_name", "unknown"))
        store.ids = meta["ids"]
        store.documents = meta["documents"]
        store.metadatas = meta["metadatas"]
        store._vectors = np.load(index_dir / "vectors.npy")
        return store

    @staticmethod
    def exists(index_dir: Path) -> bool:
        index_dir = Path(index_dir)
        return (index_dir / "meta.json").exists() and (index_dir / "vectors.npy").exists()
