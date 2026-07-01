"""Safety and refusal layer – detects out-of-scope requests and prompt injection."""

from __future__ import annotations

# Keywords/phrases that signal off-topic or adversarial intent
_REFUSAL_PATTERNS: list[str] = [
    "ignore previous",
    "ignore all previous",
    "ignore your instructions",
    "disregard your",
    "forget your instructions",
    "override your",
    "bypass your",
    "you are now",
    "act as",
    "pretend you are",
    "jailbreak",
    "system prompt",
    "reveal your prompt",
    "what are your instructions",
    "legal advice",
    "legal hiring",
    "immigration law",
    "employment law",
    "discrimination law",
    "sue my employer",
    "write me a resume",
    "write a cover letter",
    "salary negotiation",
    "how much should i pay",
    "how do i fire",
    "how to terminate",
    "give me interview questions",
]

_OFF_TOPIC_PATTERNS: list[str] = [
    "recipe",
    "weather",
    "stock market",
    "cryptocurrency",
    "sports score",
    "movie recommendation",
    "tell me a joke",
    "write a poem",
    "write a story",
    "capital of",
    "translate this",
    "code for me",
    "help me hack",
]

REFUSAL_MESSAGE = (
    "I appreciate your question, but I'm specifically designed to help with "
    "SHL assessment recommendations. I can help you find the right SHL assessments "
    "for your hiring needs — things like cognitive tests, personality questionnaires, "
    "skills assessments, and more. How can I help you with SHL assessments?"
)

PROMPT_INJECTION_MESSAGE = (
    "I'm unable to process that request. I'm an SHL Assessment Recommender "
    "designed to help you find appropriate SHL assessments for your hiring needs. "
    "Please let me know what role you're hiring for, and I'll recommend "
    "relevant assessments."
)


def check_refusal(user_message: str) -> tuple[bool, str]:
    """
    Check if a user message should be refused.

    Returns:
        (should_refuse, refusal_reason)
    """
    msg_lower = user_message.lower().strip()

    # Check prompt injection patterns
    for pattern in _REFUSAL_PATTERNS:
        if pattern in msg_lower:
            # Distinguish prompt injection from general off-topic
            if any(
                kw in pattern
                for kw in [
                    "ignore",
                    "disregard",
                    "override",
                    "bypass",
                    "forget",
                    "act as",
                    "pretend",
                    "jailbreak",
                    "system prompt",
                    "reveal",
                ]
            ):
                return True, "prompt_injection"
            return True, "off_topic"

    # Check off-topic patterns
    for pattern in _OFF_TOPIC_PATTERNS:
        if pattern in msg_lower:
            return True, "off_topic"

    return False, ""


def get_refusal_reply(reason: str) -> str:
    """Get the appropriate refusal message based on the reason."""
    if reason == "prompt_injection":
        return PROMPT_INJECTION_MESSAGE
    return REFUSAL_MESSAGE
