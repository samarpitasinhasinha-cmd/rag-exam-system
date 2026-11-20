import json
import requests
import numpy as np
import faiss
from scipy.stats import spearmanr

OLLAMA = "http://localhost:11434"
MODELS = ["nomic-embed-text", "mxbai-embed-large"]

def embed(texts, model):
    r = requests.post(f"{OLLAMA}/api/embed",
                      json={"model": model, "input": texts}, timeout=180)
    r.raise_for_status()
    return np.asarray(r.json()["embeddings"], dtype="float32")

def normalized_index(vectors):
    faiss.normalize_L2(vectors)
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    return idx

def main():
    with open("dataset.json", encoding="utf-8") as f:
        data = json.load(f)
    sentences, queries = data["sentences"], data["queries"]
    if len(sentences) != 50 or len(queries) != 10:
        raise ValueError("dataset.json must contain 50 sentences and 10 queries")

    doc_a = embed(sentences, MODELS[0])
    doc_b = embed(sentences, MODELS[1])
    idx_a, idx_b = normalized_index(doc_a), normalized_index(doc_b)

    matrix = []
    for q in queries:
        qa = embed([q], MODELS[0])
        qb = embed([q], MODELS[1])
        faiss.normalize_L2(qa)
        faiss.normalize_L2(qb)

        scores_a, ids_a = idx_a.search(qa, 50)
        scores_b, ids_b = idx_b.search(qb, 50)

        top_a = [int(x) for x in ids_a[0][:3]]
        top_b = [int(x) for x in ids_b[0][:3]]

        # Rank vectors indexed by original corpus document ID, so correlation
        # compares both models' full 50-document rankings.
        rank_a = np.empty(50, dtype=int)
        rank_b = np.empty(50, dtype=int)
        for rank, doc_id in enumerate(ids_a[0]):
            rank_a[int(doc_id)] = rank + 1
        for rank, doc_id in enumerate(ids_b[0]):
            rank_b[int(doc_id)] = rank + 1

        rho = float(spearmanr(rank_a, rank_b).statistic)
        matrix.append(top_a + top_b + [round(rho, 6)])

    print(json.dumps(matrix, indent=2))
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

if __name__ == "__main__":
    main()
