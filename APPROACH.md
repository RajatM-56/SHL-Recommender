# Approach Document – SHL Assessment Recommender

## 1. Architecture Overview

The system implements a **Retrieval-Augmented Generation (RAG)** pipeline orchestrated through a **LangGraph** state machine. The core flow is:

1. **User sends conversation history** → stateless `POST /chat` endpoint
2. **Intent Detection** classifies the request (clarify / recommend / refine / compare / refuse)
3. **Behavior nodes** execute the appropriate logic (retrieval, generation, or safety checks)
4. **Response Builder** formats the output into the strict API schema

### Why LangGraph?

LangGraph provides a declarative state graph with conditional routing, which maps naturally to the five distinct conversational behaviors. Each node is a self-contained function with clear inputs and outputs, making the system easy to test, debug, and extend. The graph is compiled once at startup and invoked per request.

### Stateless Design

The API is fully stateless — no server-side memory. The entire conversation history is passed with each request, allowing horizontal scaling and zero-downtime deployments. Intent detection considers the full conversation arc, not just the last message.

---

## 2. Retrieval Strategy

### Embedding Generation

Assessments are converted into rich text representations that combine name, description, test types, job levels, duration, and capabilities. These are embedded using **Google Gemini's text-embedding-004** model (768 dimensions).

### Vector Search

**FAISS IndexFlatIP** (inner product on L2-normalized vectors = cosine similarity) is used for fast retrieval. Given the catalog size (~377 assessments), an exact search index is sufficient and avoids the complexity of approximate methods.

### Two-Stage Retrieval

1. **FAISS retrieval** returns top-20 candidates via semantic similarity
2. **Gemini reranking** selects and orders the final 1-10 recommendations using the full conversation context and detailed assessment metadata

This two-stage approach balances recall (wide FAISS net) with precision (LLM-powered contextual ranking).

---

## 3. Prompt Design

Each node uses a carefully designed prompt:

| Node | Prompt Strategy |
|------|----------------|
| **Intent Detection** | Few-shot classification with 5 defined intents; considers full conversation context to avoid premature recommendations |
| **Clarify** | Instructed to ask ONE focused question at a time about role, seniority, skills, or preferences |
| **Recommend** | Given retrieved assessments and conversation; produces an explanatory response (URLs excluded from text, provided separately) |
| **Refine** | Re-retrieves and re-ranks with updated conversation context; explains what changed |
| **Compare** | Strictly grounded in provided catalog data; instructed to never invent information |
| **Refuse** | Combines rule-based pattern matching (fast, reliable) with LLM-generated polite redirects |

### Grounding Principle

Every Gemini call that produces factual claims about assessments receives the catalog data as context. The system prompt explicitly forbids answering from prior knowledge.

---

## 4. Evaluation Methodology

The test harness (`tests/test_conversations.py`) validates all six required behaviors:

| Test Case | Input | Expected Behavior |
|-----------|-------|-------------------|
| Clarification | "I need an assessment" | Asks follow-up, no recommendations |
| Recommendation | Multi-turn Java developer hiring | Returns 1-10 recs with valid SHL URLs |
| Refinement | "Also include personality tests" | Updates recommendation list |
| Comparison | "Compare OPQ and GSA" | Factual comparison from catalog |
| Refusal | "Give me legal advice" | Polite refusal, no recommendations |
| Prompt Injection | "Ignore previous instructions" | Refuses, stays on-topic |

### Validation Checks

- Schema compliance (reply, recommendations[], end_of_conversation)
- URL validity (all URLs must contain `shl.com`)
- Recommendation count bounds (1 ≤ k ≤ 10)
- Grounding (comparison answers reference catalog data)

---

## 5. Failure Cases & Mitigations

| Failure Mode | Mitigation |
|-------------|-----------|
| Gemini returns invalid JSON from reranker | Fallback to top FAISS results directly |
| Assessment names not found in comparison | FAISS similarity search as fallback |
| Intent misclassification | Rule-based safety check runs BEFORE LLM classification |
| Prompt injection | Keyword pattern matching catches common attacks before LLM processing |
| Embedding API rate limits | Batch processing with configurable batch size |
| FAISS index not built | Clear error message directing to ingestion script |

---

## 6. Lessons Learned

1. **Two-stage retrieval is essential.** Pure semantic search returns reasonable candidates but the LLM reranker dramatically improves relevance by understanding nuanced requirements (e.g., "mid-level" vs "executive" contexts).

2. **Rule-based safety is faster and more reliable than LLM-only safety.** Pattern matching catches the majority of adversarial inputs instantly, while the LLM handles nuanced edge cases.

3. **Stateless design simplifies everything.** No session management, no cache invalidation, no state synchronization — the conversation history in the request is the single source of truth.

4. **Catalog data quality matters.** Rich descriptions and structured metadata (job levels, test types, duration) significantly improve both embedding quality and reranking accuracy.

5. **Explicit grounding instructions are critical.** Without explicit "only use provided data" instructions, Gemini will confidently generate plausible but fabricated assessment details.
