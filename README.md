---
title: Personalized RAG API
sdk: docker
app_port: 7860
---

# Personalized RAG API

## Deploy on Hugging Face Spaces
1. Create a new Space and select **Docker**.
2. Upload `app.py`, `knowledge_base.json`, `requirements.txt`, and `Dockerfile`.
3. Wait for the Space to build and become Running.
4. Test `https://YOUR-SPACE.hf.space/health`.
5. In the exam page, enter the base URL only, without `/query`.

## API
`POST /query`
```json
{"question":"How many nodes are required for a Raft quorum?","top_k":3}
```

The response follows the required `answer`, `sources`, `scores`, and `confidence` schema.
CORS is enabled for all origins.

## Local run
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Important
The extractive QA model and embedding model are downloaded on first startup.
The first build/start can take several minutes; requests are intended to run after startup.
The QA pipeline is serialized with a lock to protect concurrent CPU inference. For higher
throughput, use a GPU or a dedicated inference worker. Validate the deployed endpoint
against the exam's 20-probe suite before saving.
