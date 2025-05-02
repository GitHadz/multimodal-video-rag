#!/usr/bin/env python
"""
Encode text & image; build FAISS, pgvector, TF‑IDF, BM25.
"""
import argparse, json, os, pickle, faiss, numpy as np, pandas as pd, torch
from pathlib import Path
import psycopg
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm, trange
import open_clip
from PIL import Image

# ───────────────────────────────── CLI flags ──────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--no-pg",   action="store_true", help="skip Postgres section")
parser.add_argument("--only-pg", action="store_true", help="run ONLY Postgres section")
args = parser.parse_args()
SKIP_PG  = args.no_pg
ONLY_PG  = args.only_pg

DATA   = Path("data")
chunks = [json.loads(l) for l in open(DATA / "chunks.jsonl")]
texts  = [c["text"] for c in chunks]

# ---------- text embeddings -----------------------
emb_path = DATA / "stella.npy"
if ONLY_PG or emb_path.exists():
    txt_emb = np.load(emb_path)
    print("🔤  Loaded existing text embeddings")
else:
    print("🔤  Encoding transcript with nomic-embed-text-v1.5 …")
    enc = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cuda")
    txt_emb = enc.encode([f"search_document: {t}" for t in texts], show_progress_bar=True, batch_size=64,
                         normalize_embeddings=True)
    np.save(emb_path, txt_emb)
    print("✓ text embeddings saved")

# ─────────────────────────────  FAISS  ────────────────────────────────
flat = faiss.IndexFlatIP(txt_emb.shape[1])
flat.add(txt_emb.astype("float32"))
faiss.write_index(flat, str(DATA / "faiss_flat.index"))

hnsw = faiss.IndexHNSWFlat(txt_emb.shape[1], 32)
hnsw.hnsw.efConstruction = 200
hnsw.add(txt_emb.astype("float32"))
faiss.write_index(hnsw, str(DATA / "faiss_hnsw.index"))
print("✓ FAISS indexes built")

# ───────────────────────────── pgvector ───────────────────────────────
if not SKIP_PG:
    try:
        with psycopg.connect(os.getenv("PG_DSN", "postgresql://postgres@localhost")) as con:
            cur = con.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS rag_chunks")
            dim = txt_emb.shape[1]
            cur.execute(f"""
                CREATE TABLE rag_chunks(
                    id INT PRIMARY KEY,
                    start FLOAT, "end" FLOAT,
                    text TEXT,
                    embedding VECTOR({dim})
                )
                """)
            for i, e in enumerate(txt_emb):
                cur.execute(
                    "INSERT INTO rag_chunks VALUES (%s,%s,%s,%s,%s)",
                    (i, chunks[i]["start"], chunks[i]["end"], texts[i], list(e))
                )
            cur.execute(
                "CREATE INDEX ON rag_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100)")
            cur.execute(
                "CREATE INDEX ON rag_chunks USING hnsw (embedding vector_cosine_ops)")
            con.commit()
        print("✓ pgvector IVFFLAT & HNSW ready")
    except Exception as e:
        print("! pgvector skipped →", e)

# ─── if we only wanted Postgres work, exit before heavy vision section ──
if ONLY_PG:
    print("ONLY_PG flag → skipping FAISS / image / lexical steps")
    raise SystemExit

# ─────────────────────── image embeddings (batched) ───────────────────
meta   = json.loads(open(DATA / "meta.json").read())
fps    = meta["fps"]
frames = sorted((DATA / "frames").glob("*.jpg"))
print(f"🖼  Encoding {len(frames)} frames with OpenCLIP …")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
model = model.to(DEVICE).eval()           # make sure weights live on the chosen device

BATCH = 64
emb_list = []
for i in trange(0, len(frames), BATCH, desc="batches"):
    imgs = [preprocess(Image.open(p).convert("RGB")) for p in frames[i:i+BATCH]]
    with torch.no_grad():
        emb = model.encode_image(torch.stack(imgs).to(DEVICE)).cpu().numpy()
    emb_list.append(emb)

img_emb = np.concatenate(emb_list)
img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)

records = [
    dict(path=str(p), ts=i / fps, vec=vec.astype("float32").tolist())
    for i, (p, vec) in enumerate(zip(frames, img_emb))
]
with open(DATA / "clip.pkl", "wb") as f:
    pickle.dump(records, f)
print("✓ image embeddings + timestamps saved")


# ─────────────────────── lexical baselines ────────────────────────────
print("📚  Building TF‑IDF & BM25 …")
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer().fit(texts)
pickle.dump(tfidf_vec, open(DATA / "tfidf.pkl", "wb"))

bm25 = BM25Okapi([t.split() for t in texts])
pickle.dump(bm25, open(DATA / "bm25.pkl", "wb"))
print("✓ lexical models stored")
