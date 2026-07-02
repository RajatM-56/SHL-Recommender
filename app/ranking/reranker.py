"""LLM-powered reranker – takes retrieved assessments and conversation context,
then selects and ranks the most relevant ones."""

from __future__ import annotations
import json
import re

from app.models.schemas import CatalogAssessment, Recommendation
from app.utils.config import settings
from app.utils.llm_client import generate_text




RERANK_PROMPT = """You are an SHL assessment recommendation engine. 
Given a conversation about hiring needs and a list of candidate SHL assessments, 
select and rank the MOST RELEVANT assessments.

CONVERSATION CONTEXT:
{conversation}

CANDIDATE ASSESSMENTS:
{assessments}

INSTRUCTIONS:
1. Analyze the conversation to understand the role, seniority, required skills, and any preferences.
2. Select the assessments that best match the requirements.
3. Rank them from most to least relevant.
4. Return between 1 and 10 assessments.
5. ONLY select assessments from the provided list. Do NOT invent assessments.

Return your answer as a JSON array of objects with these exact fields:
- "name": the exact assessment name from the list
- "url": the exact URL from the list  
- "test_type": the test category/type (from the "keys" field)

Return ONLY the JSON array, no other text. Example:
[{{"name": "Example Test", "url": "https://...", "test_type": "Knowledge & Skills"}}]
"""


def rerank_assessments(
    conversation_text: str,
    candidates: list[CatalogAssessment],
    max_results: int | None = None,
) -> list[Recommendation]:
    """Use LLM to rerank candidate assessments based on conversation context."""
    if not candidates:
        return []

    max_k = min(max_results or settings.TOP_K_FINAL, 10)

    # Build assessment summaries for the prompt
    assessment_texts = []
    for i, a in enumerate(candidates, 1):
        assessment_texts.append(
            f"{i}. Name: {a.name}\n"
            f"   URL: {a.link}\n"
            f"   Types: {', '.join(a.keys)}\n"
            f"   Job Levels: {', '.join(a.job_levels)}\n"
            f"   Duration: {a.duration or 'N/A'}\n"
            f"   Remote: {a.remote} | Adaptive: {a.adaptive}\n"
            f"   Description: {a.description[:300]}"
        )

    prompt = RERANK_PROMPT.format(
        conversation=conversation_text,
        assessments="\n\n".join(assessment_texts),
    )

    response_text = generate_text(prompt, temperature=0.1)

    # Parse JSON from response
    try:
        text = response_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        # Fallback: return top candidates directly
        return _fallback_recommendations(candidates, max_k)

    # Validate and build recommendations
    # Build a lookup of valid URLs from candidates
    valid_urls = {a.link for a in candidates}
    valid_names = {a.name.lower(): a for a in candidates}

    recommendations = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        url = item.get("url", "")
        test_type = item.get("test_type", "")

        # Validate URL is from our catalog
        if url not in valid_urls:
            # Try to find by name
            matched = valid_names.get(name.lower())
            if matched:
                url = matched.link
                if not test_type:
                    test_type = ", ".join(matched.keys)
            else:
                continue  # Skip hallucinated entries

        recommendations.append(
            Recommendation(name=name, url=url, test_type=test_type)
        )
        if len(recommendations) >= max_k:
            break

    if not recommendations:
        return _fallback_recommendations(candidates, max_k)

    return recommendations


def _fallback_recommendations(
    candidates: list[CatalogAssessment], max_k: int
) -> list[Recommendation]:
    """Simple fallback: return top candidates as-is."""
    results = []
    for a in candidates[:max_k]:
        results.append(
            Recommendation(
                name=a.name,
                url=a.link,
                test_type=", ".join(a.keys),
            )
        )
    return results
