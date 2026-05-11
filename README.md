# VARYNT – Embedding & Semantic Retrieval System

ML Engineer Final Assessment – Dream Reflection Media

## What It Does
- Accepts text input → generates 384-dim embeddings → stores in pgvector → retrieves top-k similar results using cosine similarity

## Tech Stack
- **Model:** all-MiniLM-L6-v2 (sentence-transformers)
- **DB:** PostgreSQL + pgvector (IVFFlat index)
- **API:** FastAPI
- **Cache:** Redis

## Project Structure
├── embeddings_system.py   # Core embedding + retrieval logic
├── app.py                 # FastAPI endpoints
├── evaluation.py          # Similarity quality tests
├── setup.sql              # DB setup
└── requirements.txt

## Quick Start
```bash
pip install -r requirements.txt
psql -U postgres -d varynt_db -f setup.sql
uvicorn app:app --reload
```

## API
```bash
# Store
POST /store  → { "text": "your input", "metadata": {} }

# Retrieve
POST /retrieve → { "query": "your query", "top_k": 5 }

# Health
GET /health
```


Dear Sonu,

Thank you for participating in the selection process with Dream Reflection Mediapasted
