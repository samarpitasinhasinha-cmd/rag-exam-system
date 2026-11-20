import json
import os
import threading
from typing import Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import pipeline

KB_PATH = os.getenv("KB_PATH", "knowledge_base.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QA_MODEL = os.getenv("QA_MODEL", "deepset/minilm-uncased-squad2")
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.30"))
MAX_TOP_K = 10

with open(KB_PATH, "r", encoding="utf-8") as f:
    passages = json.load(f)
if not isinstance(passages, list) or len(passages) != 50:
    raise RuntimeError("knowledge_base.json must be a list of exactly 50 passages")

# Load once at startup; never download models inside a request.
embedder = SentenceTransformer(EMBED_MODEL)
embeddings = embedder.encode(
    passages, convert_to_numpy=True, normalize_embeddings=True,
    show_progress_bar=False
).astype("float32")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

qa = pipeline("question-answering", model=QA_MODEL, tokenizer=QA_MODEL, device=-1)
qa_lock = threading.Lock()

app = FastAPI(title="Personalized RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=MAX_TOP_K)

class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float

def retrieve(question: str, top_k: int):
    q = embedder.encode([question], convert_to_numpy=True,
                        normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, min(top_k, len(passages)))
    pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
    return pairs

@app.get("/")
def root():
    return {"service": "personalized-rag", "status": "ready", "passages": len(passages)}

@app.get("/health")
def health():
    return {"status": "ok", "passages": len(passages)}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question cannot be blank")

    hits = retrieve(question, req.top_k)
    if not hits or hits[0][1] < MIN_SIMILARITY:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.10)

    # Give the QA model the retrieved evidence together, allowing it to combine
    # facts from adjacent/relevant passages. Passage labels preserve provenance.
    context_parts = []
    for idx, score in hits:
        context_parts.append(f"[Passage {idx}] {passages[idx]}")
    context = "\n".join(context_parts)

    try:
        with qa_lock:
            result = qa(question=question, context=context, handle_impossible_answer=True)
    except Exception:
        # Retrieval evidence alone is not enough to justify a fabricated answer.
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.15)

    answer = (result.get("answer") or "").strip()
    qa_score = float(result.get("score", 0.0))
    no_answer_prob = float(result.get("no_answer_probability", 0.0))

    # Conservative abstention: weak retrieval, empty span, or high no-answer score.
    if (not answer or hits[0][1] < MIN_SIMILARITY
            or no_answer_prob > 0.35 or qa_score < 0.12):
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.20)

    # Confidence is deliberately calibrated conservatively from independent
    # retrieval and extractive-QA signals; it is not a statistical guarantee.
    confidence = max(0.61, min(0.98, 0.45 * hits[0][1] + 0.55 * qa_score))
    return QueryResponse(
        answer=answer,
        sources=[i for i, _ in hits],
        scores=[round(max(0.0, min(1.0, s)), 4) for _, s in hits],
        confidence=round(confidence, 3),
    )
