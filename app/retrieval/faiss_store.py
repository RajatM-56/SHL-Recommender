"""FAISS vector store for SHL assessment retrieval."""

from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import faiss
from google import genai

from app.models.schemas import CatalogAssessment
from app.utils.config import settings


# ── Singleton client ───────────────────────────────────────────────────────

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


# ── Embedding helpers ──────────────────────────────────────────────────────


def generate_embeddings(texts: list[str], batch_size: int = 50) -> np.ndarray:
    """Generate embeddings for a list of texts using Gemini embedding API."""
    client = _get_client()
    all_embeddings = []

    import time
    from google.genai.errors import ClientError
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        max_retries = 10
        for attempt in range(max_retries):
            try:
                result = client.models.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    contents=batch,
                )
                for emb in result.embeddings:
                    all_embeddings.append(emb.values)
                break
            except ClientError as e:
                if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                    if attempt < max_retries - 1:
                        print(f"Rate limited. Sleeping for 65s (attempt {attempt+1}/{max_retries})")
                        time.sleep(65)
                    else:
                        raise
                else:
                    raise
        time.sleep(1)

    return np.array(all_embeddings, dtype=np.float32)


def generate_query_embedding(query: str) -> np.ndarray:
    """Generate embedding for a single query string."""
    client = _get_client()
    result = client.models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=[query],
    )
    return np.array(result.embeddings[0].values, dtype=np.float32).reshape(1, -1)


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

        # Normalize for cosine similarity (Inner Product on normalized = cosine)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self._loaded = True
        print(f"FAISS index built with {self.index.ntotal} vectors.")

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
        faiss.normalize_L2(query_vec)

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
