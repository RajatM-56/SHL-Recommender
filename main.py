"""SHL Assessment Recommender – FastAPI Application Entry Point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router
from app.retrieval.faiss_store import faiss_store
from app.utils.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS index on startup."""
    try:
        settings.validate()
    except ValueError as e:
        print(f"WARNING: Configuration error: {e}")
        print("WARNING: The app will start to pass health checks, but /chat may fail until configured.")
    print("Loading FAISS index...")
    faiss_store.load()
    print(f"Ready! {faiss_store.index.ntotal} assessments indexed.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational AI agent that helps hiring managers discover "
        "appropriate SHL assessments through conversation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow all origins for demo/evaluation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
