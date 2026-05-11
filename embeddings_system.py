# embeddings_system.py — Complete working system

from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

# ── MODEL LOAD ───────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast, lightweight

app = FastAPI(title="VARYNT Embedding & Retrieval System")

# ── DB CONNECTION ────────────────────────────────────────────────────
def get_db():
    return psycopg2.connect(
        host="localhost", database="varynt_db",
        user="postgres", password="password"
    )

# ── DB SETUP ─────────────────────────────────────────────────────────
def setup_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lead_embeddings (
            id SERIAL PRIMARY KEY,
            user_input TEXT NOT NULL,
            embedding vector(384),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS embedding_idx 
        ON lead_embeddings 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("DB setup complete")

# ── INPUT MODELS ─────────────────────────────────────────────────────
class InputRequest(BaseModel):
    text: str
    metadata: dict = {}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# ── CORE FUNCTIONS ───────────────────────────────────────────────────
def generate_embedding(text: str) -> list:
    """Generate 384-dim embedding from text."""
    if not text or len(text.strip()) < 3:
        raise ValueError("Input text too short")
    embedding = model.encode(text.strip(), normalize_embeddings=True)
    return embedding.tolist()

def store_embedding(text: str, embedding: list, metadata: dict = {}) -> int:
    """Store text + embedding in pgvector."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO lead_embeddings (user_input, embedding, metadata)
               VALUES (%s, %s, %s) RETURNING id""",
            (text, embedding, json.dumps(metadata))
        )
        record_id = cur.fetchone()[0]
        conn.commit()
        return record_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def retrieve_similar(query_embedding: list, top_k: int = 5) -> list:
    """Retrieve top-k most similar records using cosine similarity."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """SELECT id, user_input, metadata,
               1 - (embedding <=> %s::vector) AS similarity
               FROM lead_embeddings
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (query_embedding, query_embedding, top_k)
        )
        results = cur.fetchall()
        return [
            {
                "id": r[0],
                "text": r[1],
                "metadata": r[2],
                "similarity_score": round(float(r[3]), 4)
            }
            for r in results
        ]
    finally:
        cur.close()
        conn.close()

# ── API ENDPOINTS ─────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    setup_db()

@app.post("/store")
def store_input(req: InputRequest):
    """Accept input → generate embedding → store."""
    if not req.text or len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Input too short or empty")

    try:
        start = time.time()
        embedding = generate_embedding(req.text)
        record_id = store_embedding(req.text, embedding, req.metadata)
        latency = round(time.time() - start, 3)

        return {
            "status": "stored",
            "id": record_id,
            "embedding_dim": len(embedding),
            "latency_seconds": latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")

@app.post("/retrieve")
def retrieve(req: QueryRequest):
    """Accept query → generate embedding → retrieve similar."""
    if not req.query or len(req.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")

    if req.top_k < 1 or req.top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")

    try:
        start = time.time()
        query_embedding = generate_embedding(req.query)
        results = retrieve_similar(query_embedding, req.top_k)
        latency = round(time.time() - start, 3)

        return {
            "query": req.query,
            "results": results,
            "count": len(results),
            "latency_seconds": latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "model": "all-MiniLM-L6-v2", "dim": 384}
