#!/usr/bin/env python
"""
Compute accuracy@1, rejection, and mean latency for every retrieval mode.
"""
import json, time, yaml, numpy as np, sys
from query import search

gold = yaml.safe_load(open("gold/qas.yaml"))
ANS  = gold["answerable"];  UNANS = gold["unanswerable"]

def run(mode):
    ok = rej = 0; t = []
    for qa in ANS:
        tic = time.time(); hits = search(qa["q"], mode=mode)
        t.append(time.time()-tic)
        if hits and abs(hits[0][1]["start"] - qa["ts"]) < 5:
            ok += 1
    for q in UNANS:
        tic = time.time(); hits = search(q, mode=mode)
        t.append(time.time()-tic)
        if not hits or hits[0][0] < .50:
            rej += 1
    return dict(acc=ok/len(ANS), rej=rej/len(UNANS),
                lat=np.mean(t))

modes = ["faiss-flat", "faiss-hnsw",
         "pg-ivf", "pg-hnsw",
         "bm25", "tfidf"]

for m in modes:
    print(m.ljust(10), run(m))
