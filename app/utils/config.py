"""SHL Assessment Recommender - Configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    # Local sentence-transformers model for embeddings (no API needed)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    # Generation model runs on Groq
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    CATALOG_PATH: str = os.getenv("CATALOG_PATH", "data/catalog.json")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Embedding dimension for all-MiniLM-L6-v2
    EMBEDDING_DIM: int = 384

    # Retrieval settings
    TOP_K_RETRIEVAL: int = 20  # Retrieve more, then rerank
    TOP_K_FINAL: int = 10  # Max recommendations returned

    def validate(self) -> None:
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Check your .env file.")


settings = Settings()
