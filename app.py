import streamlit as st, json, time
from pipelines.query import search

VIDEO_URL = "https://www.youtube.com/watch?v=dARr3lGKwk8"
st.set_page_config(page_title="Multimodal Video RAG", layout="wide")
st.title("🎥 Ask the Lecture")

if "history" not in st.session_state:
    st.session_state.history = []

algo = st.sidebar.selectbox(
    "Retrieval mode",
    ["faiss-flat", "faiss-hnsw", "pg-ivf", "pg-hnsw", "bm25", "tfidf"]
)

query = st.chat_input("Ask anything about the video")
if query:
    start = time.perf_counter()
    results = search(query, k=3, mode=algo)
    latency = time.perf_counter() - start

    if results and results[0][0] > 0.25:    # acceptance threshold
        best = results[0][1]
        ts = int(best["start"])
        st.session_state.history.append(
            (query, f"⏱ Found at **{ts//60}:{ts%60:02d}**  \n\n> {best['text']} …")
        )
    else:
        st.session_state.history.append(
            (query, "🚫 I’m pretty sure that topic never comes up in this video.")
        )
    st.toast(f"{algo} answered in {latency*1e3:.1f} ms")

# ---------- chat history ----------
for q, a in st.session_state.history[::-1]:
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
        if a.startswith("⏱"):
            ts = int(a.split("**")[1].split("**")[0].split(":")[0])*60 + \
                 int(a.split(":")[2].split("**")[0])
            st.video(VIDEO_URL + f"&t={ts}s")
