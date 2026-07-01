#!/usr/bin/env python3
"""Catalog Ingestion Script – builds the FAISS index from the SHL catalog JSON."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.catalog.scraper import load_catalog
from app.retrieval.faiss_store import FAISSStore
from app.utils.config import settings


def main():
    print("=" * 60)
    print("SHL Catalog Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load catalog
    print(f"\n[1/3] Loading catalog from {settings.CATALOG_PATH}...")
    assessments = load_catalog()
    print(f"  → Loaded {len(assessments)} assessments.")

    # Step 2: Build FAISS index
    print(f"\n[2/3] Building FAISS index...")
    store = FAISSStore()
    store.build_index(assessments)

    # Step 3: Save index
    print(f"\n[3/3] Saving FAISS index to {settings.FAISS_INDEX_PATH}...")
    store.save()

    print(f"\n{'=' * 60}")
    print("Ingestion complete!")
    print(f"  Total assessments indexed: {store.index.ntotal}")
    print(f"  Index location: {settings.FAISS_INDEX_PATH}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
