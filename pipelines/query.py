"""
Unified search API used by the UI and evaluation.
Modes: faiss-flat | faiss-hnsw | pg-ivf | pg-hnsw | bm25 | tfidf
"""
import json, pickle, os
from pathlib import Path

import faiss, numpy as np, psycopg
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data")
chunks = [json.loads(l) for l in open(DATA/"chunks.jsonl")]

# ---------- load embeddings & models -------------
TXT_EMB = np.load(DATA/"stella.npy")
enc      = SentenceTransformer("WhereIsAI/UAE-Large-V1")
bm25     = pickle.load(open(DATA/"bm25.pkl", "rb"))
tfidf_vec= pickle.load(open(DATA/"tfidf.pkl", "rb"))

FAISS = {
    "flat": faiss.read_index(str(DATA/"faiss_flat.index")),
    "hnsw": faiss.read_index(str(DATA/"faiss_hnsw.index")),
}

texts = [c["text"] for c in chunks]        # for TF‑IDF cosine

# ---------- helpers ------------------------------
def _dedup(hits):
    seen = {}
    for score, idx in hits:
        key = round(chunks[idx]["start"]/5)  # merge by 5‑s bucket
        if key not in seen or score > seen[key][0]:
            seen[key] = (score, idx)
    return sorted(seen.values(), reverse=True)

# ---------- public search ------------------------
def search(q, k=5, mode="faiss-flat"):
    """
    Return list[(score, chunk_dict)]
    """
    if mode.startswith("faiss"):
        vec  = enc.encode([q], normalize_embeddings=True)[0]
        idx  = FAISS["flat" if "flat" in mode else "hnsw"]
        sims, ids = idx.search(vec[None, :].astype("float32"), k*3)
        hits = list(zip(sims[0], ids[0]))

    elif mode.startswith("pg"):
        algo = "ivf" if "ivf" in mode else "hnsw"
        vec  = enc.encode([q])[0].tolist()
        with psycopg.connect(os.getenv("PG_DSN", "postgresql://postgres@localhost")) as con:
            cur = con.cursor()
            cur.execute(f"""
              SELECT id, 1 - (embedding <=> %s) AS s
              FROM rag_chunks
              ORDER BY embedding <=> %s
              LIMIT %s
            """, (vec, vec, k*3))
            hits = [(s, idx) for idx, s in cur.fetchall()]

    elif mode == "bm25":
        scores = bm25.get_scores(q.split())
        ids    = scores.argsort()[::-1][:k*3]
        hits   = list(zip(scores[ids], ids))

    elif mode == "tfidf":
        q_vec  = tfidf_vec.transform([q])
        sims   = (q_vec @ tfidf_vec.T).toarray()[0]
        ids    = sims.argsort()[::-1][:k*3]
        hits   = list(zip(sims[ids], ids))

    else:
        raise ValueError("unknown mode")

    best = _dedup(hits)[:k]
    return [(s, chunks[i]) for s, i in best]
