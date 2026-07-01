"""Catalog loader – reads the scraped SHL JSON and returns structured assessments."""

from __future__ import annotations
import json
from pathlib import Path

from app.models.schemas import CatalogAssessment
from app.utils.config import settings


def load_catalog(path: str | None = None) -> list[CatalogAssessment]:
    """Load and normalize the SHL catalog JSON into CatalogAssessment objects."""
    catalog_path = Path(path or settings.CATALOG_PATH)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    assessments: list[CatalogAssessment] = []
    for item in raw:
        # Skip items with bad status
        if item.get("status") != "ok":
            continue

        assessment = CatalogAssessment(
            entity_id=str(item.get("entity_id", "")),
            name=item.get("name", ""),
            link=item.get("link", ""),
            description=item.get("description", ""),
            job_levels=item.get("job_levels", []),
            languages=item.get("languages", []),
            duration=item.get("duration", ""),
            remote=item.get("remote", ""),
            adaptive=item.get("adaptive", ""),
            keys=item.get("keys", []),
        )
        assessments.append(assessment)

    return assessments


def get_catalog_map(assessments: list[CatalogAssessment]) -> dict[str, CatalogAssessment]:
    """Create a lookup map: entity_id -> CatalogAssessment."""
    return {a.entity_id: a for a in assessments}


def find_assessments_by_name(
    assessments: list[CatalogAssessment],
    names: list[str],
) -> list[CatalogAssessment]:
    """Fuzzy-find assessments by name substrings (for compare intent)."""
    results = []
    for name_query in names:
        query_lower = name_query.lower().strip()
        for a in assessments:
            if query_lower in a.name.lower():
                results.append(a)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for a in results:
        if a.entity_id not in seen:
            seen.add(a.entity_id)
            unique.append(a)
    return unique
