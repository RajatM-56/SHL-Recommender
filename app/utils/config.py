"""SHL Assessment Recommender - Configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gemini-2.0-flash")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    CATALOG_PATH: str = os.getenv("CATALOG_PATH", "data/catalog.json")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Embedding dimension for text-embedding-004
    EMBEDDING_DIM: int = 768

    # Retrieval settings
    TOP_K_RETRIEVAL: int = 20  # Retrieve more, then rerank
    TOP_K_FINAL: int = 10  # Max recommendations returned

    def validate(self) -> None:
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")


settings = Settings()
