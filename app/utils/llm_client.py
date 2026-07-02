"""Unified LLM client with retry logic.

Uses Groq for text generation. Embeddings are handled locally by sentence-transformers.
Includes exponential backoff for rate-limit resilience.
"""

from __future__ import annotations

import time
import logging

from groq import Groq, RateLimitError, APIError

from app.utils.config import settings

logger = logging.getLogger(__name__)

# ── Singleton clients ──────────────────────────────────────────────────────

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


# ── Text generation via Groq ──────────────────────────────────────────────


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    max_retries: int = 5,
) -> str:
    """Generate text using Groq with automatic retry on rate limits.

    Args:
        prompt: The text prompt to send.
        model: Override the default generation model.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        max_retries: Number of retries on rate-limit errors.

    Returns:
        The generated text string.
    """
    client = _get_groq_client()
    model = model or settings.GENERATION_MODEL

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            wait = min(2 ** attempt * 5, 60)  # 5s, 10s, 20s, 40s, 60s
            logger.warning(
                "Groq rate limited (attempt %d/%d). Waiting %ds...",
                attempt + 1, max_retries, wait,
            )
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise

        except APIError as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = min(2 ** attempt * 5, 60)
                logger.warning(
                    "Groq API error (attempt %d/%d): %s. Waiting %ds...",
                    attempt + 1, max_retries, e, wait,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    raise
            else:
                raise

    # Should not reach here, but just in case
    raise RuntimeError("generate_text: all retries exhausted")
