import json, os, re
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding

KB_PATH = os.getenv("KB_PATH", "knowledge_base.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.30"))
with open(KB_PATH, encoding="utf-8") as f:
    passages = json.load(f)
if not isinstance(passages, list) or len(passages) != 50:
    raise RuntimeError("knowledge_base.json must contain exactly 50 passages")

# ONNX-based embeddings avoid the much larger PyTorch runtime.
embedder = TextEmbedding(model_name=EMBED_MODEL)
doc_vectors = np.asarray(list(embedder.embed(passages)), dtype=np.float32)
doc_vectors /= np.maximum(np.linalg.norm(doc_vectors, axis=1, keepdims=True), 1e-12)

app = FastAPI(title="Personalized RAG API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float

def answer_sentence(question, passage):
    sentences = re.split(r"(?<=[.!?])\\s+", passage.strip())
    q = set(re.findall(r"[a-z0-9]+", question.lower()))
    def overlap(s):
        words = set(re.findall(r"[a-z0-9]+", s.lower()))
        return len(q & words) / max(1, len(q)) + 0.05 * len(q & words)
    return max(sentences or [passage], key=overlap)

@app.get("/")
def root():
    return {"service":"personalized-rag", "status":"ready", "passages":len(passages)}

@app.get("/health")
def health():
    return {"status":"ok", "passages":len(passages)}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question cannot be blank")
    qvec = np.asarray(list(embedder.embed([question])), dtype=np.float32)[0]
    qvec /= max(float(np.linalg.norm(qvec)), 1e-12)
    sims = doc_vectors @ qvec
    ids = np.argsort(-sims)[:req.top_k]
    hits = [(int(i), float(sims[i])) for i in ids]
    if not hits or hits[0][1] < MIN_SIMILARITY:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.10)
    best_id, best_score = hits[0]
    answer = answer_sentence(question, passages[best_id])
    confidence = min(0.95, max(0.61, 0.55 + (best_score-MIN_SIMILARITY)*0.8))
    return QueryResponse(answer=answer, sources=[i for i,_ in hits],
        scores=[round(max(0.0,min(1.0,s)),4) for _,s in hits],
        confidence=round(confidence,3))
