#!/usr/bin/env python
"""
Encode text & image; build FAISS, pgvector, TF‑IDF, BM25.
"""
import json, os, pickle, faiss, numpy as np, pandas as pd, torch
from pathlib import Path
import psycopg
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import open_clip

DATA = Path("data")
chunks = [json.loads(l) for l in open(DATA/"chunks.jsonl")]
texts  = [c["text"] for c in chunks]

# ---------- text embeddings -----------------------
print("🔤  Encoding transcript with UAE‑Large‑V1 …")
enc = SentenceTransformer("WhereIsAI/UAE-Large-V1")
txt_emb = enc.encode(texts, show_progress_bar=True, batch_size=64,
                     normalize_embeddings=True)
np.save(DATA/"stella.npy", txt_emb)

# FAISS flat & HNSW
flat = faiss.IndexFlatIP(txt_emb.shape[1]); flat.add(txt_emb.astype("float32"))
faiss.write_index(flat, DATA/"faiss_flat.index")

hnsw = faiss.IndexHNSWFlat(txt_emb.shape[1], 32)
hnsw.hnsw.efConstruction = 200; hnsw.add(txt_emb.astype("float32"))
faiss.write_index(hnsw, DATA/"faiss_hnsw.index")
print("✓ FAISS indexes built")

# ---------- pgvector ------------------------------
try:
    with psycopg.connect(os.getenv("PG_DSN", "postgresql://postgres@localhost")) as con:
        cur = con.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS rag_chunks")
        cur.execute("""
          CREATE TABLE rag_chunks(
            id INT PRIMARY KEY,
            start FLOAT, "end" FLOAT,
            text TEXT,
            embedding VECTOR(768)
          )
        """)
        for i, e in enumerate(txt_emb):
            cur.execute(
                "INSERT INTO rag_chunks VALUES (%s,%s,%s,%s,%s)",
                (i, chunks[i]["start"], chunks[i]["end"],
                 texts[i], list(e))
            )
        cur.execute(
            "CREATE INDEX ON rag_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100)")
        cur.execute(
            "CREATE INDEX ON rag_chunks USING hnsw (embedding vector_cosine_ops)")
        con.commit()
    print("✓ pgvector IVFFLAT & HNSW ready")
except Exception as e:
    print("! pgvector skipped →", e)

# ---------- image embeddings ----------------------
meta = json.loads(open("data/meta.json").read()); fps = meta["fps"]
frames = sorted((DATA/"frames").glob("*.jpg"))
print("🖼  Encoding", len(frames), "frames with OpenCLIP …")

model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
model.eval()

imgs = torch.stack([preprocess(open_clip.load_image(str(p))) for p in frames])
with torch.no_grad():
    img_emb = model.encode_image(imgs).cpu().numpy()
img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)

records = [
    dict(path=str(p), ts=i/fps, vec=vec.astype("float32").tolist())
    for i, (p, vec) in enumerate(zip(frames, img_emb))
]
with open(DATA/"clip.pkl", "wb") as f: pickle.dump(records, f)
print("✓ image embeddings + timestamps saved")

# ---------- lexical corpora -----------------------
print("📚  Building TF‑IDF & BM25 …")
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer().fit(texts)
pickle.dump(tfidf_vec, open(DATA/"tfidf.pkl", "wb"))

bm25 = BM25Okapi([t.split() for t in texts])
pickle.dump(bm25, open(DATA/"bm25.pkl", "wb"))
print("✓ lexical models stored")
