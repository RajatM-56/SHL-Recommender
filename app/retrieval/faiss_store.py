"""FAISS vector store for SHL assessment retrieval.

Uses sentence-transformers for local embeddings (no API calls, no rate limits).
"""

from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import faiss

from app.models.schemas import CatalogAssessment
from app.utils.config import settings


# ── Singleton embedding model ──────────────────────────────────────────────

_model: 'SentenceTransformer' | None = None


def _get_model() -> 'SentenceTransformer':
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


# ── Embedding helpers ──────────────────────────────────────────────────────


def generate_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Generate embeddings for a list of texts using sentence-transformers (local)."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Pre-normalize for cosine similarity
    )
    return np.array(embeddings, dtype=np.float32)


def generate_query_embedding(query: str) -> np.ndarray:
    """Generate embedding for a single query string."""
    model = _get_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )
    return np.array(embedding, dtype=np.float32)


# ── FAISS Index ────────────────────────────────────────────────────────────


class FAISSStore:
    """FAISS vector index over the SHL catalog."""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.assessments: list[CatalogAssessment] = []
        self._loaded = False

    def build_index(self, assessments: list[CatalogAssessment]) -> None:
        """Build FAISS index from a list of assessments."""
        self.assessments = assessments
        texts = [a.to_embedding_text() for a in assessments]

        print(f"Generating embeddings for {len(texts)} assessments...")
        embeddings = generate_embeddings(texts)

        # Already normalized via normalize_embeddings=True
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self._loaded = True
        print(f"FAISS index built with {self.index.ntotal} vectors (dim={embeddings.shape[1]}).")

    def save(self, path: str | None = None) -> None:
        """Save FAISS index and metadata to disk."""
        save_dir = Path(path or settings.FAISS_INDEX_PATH)
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        with open(save_dir / "assessments.pkl", "wb") as f:
            pickle.dump(self.assessments, f)
        print(f"Index saved to {save_dir}")

    def load(self, path: str | None = None) -> None:
        """Load FAISS index and metadata from disk."""
        load_dir = Path(path or settings.FAISS_INDEX_PATH)
        index_path = load_dir / "index.faiss"
        meta_path = load_dir / "assessments.pkl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {load_dir}. "
                "Run `python scripts/ingest_catalog.py` first."
            )

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.assessments = pickle.load(f)
        self._loaded = True
        print(f"FAISS index loaded: {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int | None = None) -> list[CatalogAssessment]:
        """Search the FAISS index and return the top-k matching assessments."""
        if not self._loaded:
            self.load()

        k = min(top_k or settings.TOP_K_RETRIEVAL, self.index.ntotal)
        query_vec = generate_query_embedding(query)
        # Already normalized via normalize_embeddings=True

        scores, indices = self.index.search(query_vec, k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.assessments):
                results.append(self.assessments[idx])
        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ── Module-level singleton ─────────────────────────────────────────────────

faiss_store = FAISSStore()
