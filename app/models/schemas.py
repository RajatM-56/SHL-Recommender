"""Pydantic models for API request/response and internal data."""

from __future__ import annotations
from pydantic import BaseModel, Field


# ── API Schemas (strict contract from assignment) ──────────────────────────


class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    """POST /chat request body."""
    messages: list[ChatMessage] = Field(..., description="Full conversation history")


class Recommendation(BaseModel):
    """A single assessment recommendation."""
    name: str = Field(..., description="Assessment name from the SHL catalog")
    url: str = Field(..., description="URL to the assessment in the SHL catalog")
    test_type: str = Field(..., description="Category/type of the test")


class ChatResponse(BaseModel):
    """POST /chat response body."""
    reply: str = Field(..., description="Assistant's textual response")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="1-10 recommendations when recommending, else empty",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the task is fully complete",
    )


class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str = "ok"


# ── Internal Data Schemas ──────────────────────────────────────────────────


class CatalogAssessment(BaseModel):
    """Structured representation of one SHL assessment from the catalog."""
    entity_id: str
    name: str
    link: str
    description: str = ""
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    duration: str = ""
    remote: str = ""
    adaptive: str = ""
    keys: list[str] = Field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Create a rich text representation for embedding generation."""
        parts = [
            f"Assessment: {self.name}",
            f"Description: {self.description}",
            f"Test Types: {', '.join(self.keys)}",
            f"Job Levels: {', '.join(self.job_levels)}",
            f"Duration: {self.duration}" if self.duration else "",
            f"Remote Testing: {self.remote}",
            f"Adaptive/IRT: {self.adaptive}",
        ]
        return "\n".join(p for p in parts if p)

    def to_search_summary(self) -> str:
        """Short summary for LLM context."""
        return (
            f"• {self.name} | Types: {', '.join(self.keys)} | "
            f"Levels: {', '.join(self.job_levels[:3])}{'...' if len(self.job_levels) > 3 else ''} | "
            f"Duration: {self.duration or 'N/A'} | "
            f"Remote: {self.remote} | Adaptive: {self.adaptive}\n"
            f"  URL: {self.link}\n"
            f"  {self.description[:200]}{'...' if len(self.description) > 200 else ''}"
        )


# ── Graph State ────────────────────────────────────────────────────────────


class GraphState(BaseModel):
    """State passed through the LangGraph workflow."""
    messages: list[ChatMessage] = Field(default_factory=list)
    intent: str = ""  # clarify | recommend | refine | compare | refuse
    retrieved_assessments: list[CatalogAssessment] = Field(default_factory=list)
    reply: str = ""
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False
