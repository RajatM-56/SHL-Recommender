# SHL Assessment Recommender

A conversational AI agent that helps hiring managers and recruiters discover appropriate SHL assessments through natural conversation. Built with **FastAPI**, **LangGraph**, **Google Gemini**, and **FAISS**.

## Features

- **Conversational Interface**: Engage in natural dialogue to discover the right assessments
- **Smart Clarification**: Asks follow-up questions when requirements are vague
- **RAG-Powered Recommendations**: Semantic search + Gemini reranking for accurate results
- **Refinement**: Update recommendations as requirements change
- **Comparison**: Compare SHL assessments using catalog data only (no hallucination)
- **Safety Layer**: Refuses off-topic requests, legal advice, and prompt injection

## Architecture

```
START
  │
  ▼
Intent Detection (Gemini + rules)
  │
  ├──► Clarify    → Ask follow-up questions
  ├──► Recommend  → FAISS retrieval + Gemini reranking
  ├──► Refine     → Update recommendations
  ├──► Compare    → Catalog-grounded comparison
  └──► Refuse     → Safety refusal
  │
  ▼
Response Builder
  │
 END
```

## Quick Start

### Prerequisites

- Python 3.12+
- A Google Gemini API key

### 1. Clone & Install

```bash
git clone <repo-url>
cd SHLAssessment

python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Build the FAISS Index

```bash
python scripts/ingest_catalog.py
```

This reads `data/catalog.json`, generates Gemini embeddings for all 377 assessments, and saves the FAISS index to `data/faiss_index/`.

### 4. Run the Server

```bash
python main.py
# Or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer"}
    ]
  }'
```

## API Reference

### `GET /health`

```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need an assessment"},
    {"role": "assistant", "content": "What role are you hiring for?"},
    {"role": "user", "content": "A mid-level data analyst with SQL skills"}
  ]
}
```

**Response:**
```json
{
  "reply": "Based on your requirements...",
  "recommendations": [
    {
      "name": "SQL Server (New)",
      "url": "https://www.shl.com/products/product-catalog/view/sql-server-new/",
      "test_type": "Knowledge & Skills"
    }
  ],
  "end_of_conversation": false
}
```

## Docker Deployment

```bash
# Build
docker build -t shl-recommender .

# Run
docker run -p 8000:8000 --env-file .env shl-recommender
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
SHLAssessment/
├── app/
│   ├── api/          # FastAPI endpoints
│   ├── agents/       # (reserved for future agent extensions)
│   ├── graph/        # LangGraph workflow
│   ├── retrieval/    # FAISS vector store
│   ├── ranking/      # Gemini reranker
│   ├── catalog/      # Catalog loader/parser
│   ├── safety/       # Refusal & safety layer
│   ├── models/       # Pydantic schemas
│   └── utils/        # Configuration
├── tests/            # Evaluation harness
├── scripts/          # Ingestion pipeline
├── data/             # Catalog JSON & FAISS index
├── notebooks/        # (exploratory notebooks)
├── main.py           # Entry point
├── requirements.txt
├── Dockerfile
├── .env.example
└── APPROACH.md       # Design document
```
