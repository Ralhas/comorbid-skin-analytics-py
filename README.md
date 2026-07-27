# Comorbid Skin Analytics — v3 (Python Service)

Production service layer for the ingredient-conflict analysis system originally built in R. This repo picks up where the R-based v1/v2 work left off: the analysis and RAG logic are ported to Python and wrapped as a tested, containerized API.

The earlier R/Shiny version (data analysis, clustering, ingredient intelligence, deterministic RAG) lives at [comorbid-skin-analytics-R](https://github.com/Ralhas/comorbid-skin-analytics-R). That repo has the statistical analysis and the reasoning behind the deterministic-retrieval design. This one is the production version of the RAG piece.

## What changed from R to Python

The retrieval logic (condition/ingredient extraction, context building) was ported line-for-line from `R/09_rag_context.R` to `rag_context.py`, then verified against the same test queries to confirm identical behavior. Everything downstream — the API, the vector search, the structured outputs, the evals — is new in this version.

## Architecture
## Stack

- **FastAPI** — HTTP service layer
- **PostgreSQL + pgvector** — semantic search fallback for indirect phrasing (e.g. "vitamin B3" -> niacinamide)
- **Gemini API** — embeddings (`gemini-embedding-001`) and structured generation
- **Docker** — containerized, reproducible environment
- **pytest** — deterministic logic tests, run in CI without API calls
- **GitHub Actions** — tests + Docker build on every push

## Why deterministic retrieval + structured LLM output

The retrieval step never lets the LLM decide risk levels — it only summarizes records already found by exact or vector matching. The LLM's output is constrained to a fixed JSON schema (`risk_level`, `mechanism`, `summary`, `confidence`), and it's told to return `unknown`/`low confidence` instead of guessing when there's no match. An 8-case eval set (`evals.py`) checks this against known ingredients, known combinations, and one intentionally irrelevant question — 8/8 correct, including the irrelevant case coming back as `unknown` instead of a made-up answer.

## Environment variables

Copy `.env.example` to `.env` and fill in your own Gemini API key:

```bash
cp .env.example .env
```

`.env` is already excluded via `.gitignore` and never gets committed.

## Running locally

```bash
# 1. Start the vector DB
docker run --name pgvector-db -e POSTGRES_PASSWORD=comorbid123 \
  -e POSTGRES_DB=comorbid -p 5432:5432 -d pgvector/pgvector:pg16

# 2. Set your API key (or use the .env file above)
export GEMINI_API_KEY="your-key-here"

# 3. Install dependencies
pip install -r requirements.txt

# 4. One-time: build the embeddings table
python setup_pgvector.py

# 5. Run the service
uvicorn main:app --reload
```

Or with Docker:

```bash
docker build -t comorbid-api .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key-here" comorbid-api
```

Then open `http://localhost:8000/docs` to try the `/ask` endpoint.

## Testing

```bash
python -m pytest tests/ -v      # deterministic logic, no API cost
python evals.py                  # full LLM eval set, requires GEMINI_API_KEY
```

## Status

This is v3 of a four-part series. v4 adds LangGraph-based orchestration and an MCP interface.
