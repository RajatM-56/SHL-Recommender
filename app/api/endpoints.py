"""FastAPI endpoints for the SHL Assessment Recommender."""

from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    GraphState,
    Recommendation,
)
from app.graph.workflow import agent_graph

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless chat endpoint.
    Takes full conversation history, runs the LangGraph workflow, returns response.
    """
    try:
        # Build initial graph state from request
        initial_state = GraphState(messages=request.messages)

        # Run the LangGraph workflow
        result = agent_graph.invoke(initial_state.model_dump())

        # Extract response fields
        reply = result.get("reply", "I'm sorry, I encountered an issue. Please try again.")
        recommendations_raw = result.get("recommendations", [])
        end_of_conversation = result.get("end_of_conversation", False)

        # Parse recommendations
        recommendations = []
        for r in recommendations_raw:
            if isinstance(r, dict):
                recommendations.append(Recommendation(**r))
            elif isinstance(r, Recommendation):
                recommendations.append(r)

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation,
        )
    except Exception as e:
        # Return a graceful error message inside the ChatResponse schema 
        # so the client doesn't get a raw 500 error, especially for API rate limits.
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            reply = "I'm currently experiencing heavy traffic and hit a rate limit with the Google Gemini API. Please wait about a minute and try again!"
        else:
            reply = f"An internal error occurred: {error_msg}"
            
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )
