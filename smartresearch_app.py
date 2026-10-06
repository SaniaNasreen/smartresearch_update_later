import math
import re
import sqlite3
import time
from datetime import datetime

import arxiv
import faiss
import nltk
import numpy as np
import pandas as pd
import spacy
import streamlit as st
from bertopic import BERTopic
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# ---------------------------
# SETUP
# ---------------------------
st.set_page_config(page_title="SmartResearch Advisor", layout="centered")


@st.cache_resource(show_spinner="Loading NLP models...")
def load_models():
    nltk.download("stopwords", quiet=True)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return nlp, set(stopwords.words("english")), emb_model


nlp, STOP, EMB_MODEL = load_models()

DOMAINS = {
    "Artificial Intelligence": "artificial intelligence",
    "Web Development": "web development",
    "Renewable Energy": "renewable energy",
}
LEVELS = ["Beginner", "Intermediate", "Advanced"]


# ---------------------------
# DATA FUNCTIONS
# ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_arxiv(query="artificial intelligence", max_results=150):
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    rows = []
    for r in client.results(search):
        rows.append({
            "title": r.title,
            "abstract": r.summary,
            "url": r.entry_id,
            "published": r.published.date().isoformat(),
        })
    return pd.DataFrame(rows)


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def normalize(doc: str) -> str:
    tokens = []
    for t in nlp(doc):
        if t.is_stop or t.is_punct or t.like_num:
            continue
        lemma = t.lemma_.lower()
        if lemma and lemma not in STOP and len(lemma) > 2:
            tokens.append(lemma)
    return " ".join(tokens)


def prepare_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).apply(clean_text)
    df["norm"] = df["text"].apply(normalize)
    return df


# ---------------------------
# EMBEDDINGS + FAISS
# ---------------------------
class VectorIndex:
    def __init__(self, texts):
        self.texts = texts
        self.emb = EMB_MODEL.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)

    def search(self, query, k=20):
        k = min(k, len(self.texts))
        q = EMB_MODEL.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, ids = self.index.search(q, k)
        # faiss pads with -1 when fewer than k results exist
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


# ---------------------------
# TOPIC GENERATION
# ---------------------------
def generate_topics(norm_texts):
    if len(norm_texts) < 10:
        return []
    topic_model = BERTopic(
        min_topic_size=5,  # 30 seed papers is too few for min_topic_size=15
        calculate_probabilities=False,
        verbose=False,
    )
    topic_model.fit_transform(norm_texts)
    info = topic_model.get_topic_info()
    suggestions = []
    for _, row in info.iterrows():
        if row.Topic == -1:  # skip the outlier bucket
            continue
        words = [w for w, _ in (topic_model.get_topic(row.Topic) or [])][:5]
        if words:
            suggestions.append("Exploring: " + ", ".join(words))
        if len(suggestions) >= 8:
            break
    return suggestions


# ---------------------------
# RANKING (recency x diversity x difficulty)
# ---------------------------
def recency_weight(pub_date, tau=180):
    try:
        days_old = (datetime.now().date() - datetime.fromisoformat(pub_date).date()).days
        return math.exp(-max(days_old, 0) / tau)
    except (TypeError, ValueError):
        return 1.0


def mmr_select(candidates_emb, query_emb, k=5, lambda_=0.7):
    chosen = []
    cand_ids = list(range(len(candidates_emb)))
    sims_to_query = (candidates_emb @ query_emb.T).flatten()
    while cand_ids and len(chosen) < k:
        best_score, best = None, None
        for i in cand_ids:
            div = max((float(candidates_emb[i] @ candidates_emb[j]) for j in chosen), default=0.0)
            mmr = lambda_ * sims_to_query[i] - (1 - lambda_) * div
            if best_score is None or mmr > best_score:
                best_score, best = mmr, i
        chosen.append(best)
        cand_ids.remove(best)
    return chosen


def difficulty_weight(abstract, level):
    n = len(abstract.split())
    if level == "Beginner":
        return 1.0 if n < 120 else 0.7
    if level == "Advanced":
        return 1.0 if n > 100 else 0.8
    return 1.0


def rank_topics(seed_df, query, level):
    emb = EMB_MODEL.encode(seed_df["norm"].tolist(), normalize_embeddings=True, convert_to_numpy=True)
    q = EMB_MODEL.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

    top_ids = mmr_select(emb, q, k=5)  # diverse top-5
    ranked = seed_df.iloc[top_ids].copy()
    ranked["recency"] = ranked["published"].apply(recency_weight)
    ranked["score"] = ranked["recency"] * ranked["abstract"].apply(lambda a: difficulty_weight(a, level))
    return ranked.sort_values("score", ascending=False)


# ---------------------------
# FEEDBACK (SQLite)
# ---------------------------
DB_PATH = "feedback.db"


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS feedback(
                ts INTEGER, domain TEXT, level TEXT, title TEXT, vote INTEGER
            )"""
        )


def save_feedback(domain, level, title, vote):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO feedback VALUES(?,?,?,?,?)",
            (int(time.time()), domain, level, title, vote),
        )


init_db()


# ---------------------------
# PIPELINE
# ---------------------------
def run_pipeline(domain_label, level):
    query = DOMAINS[domain_label]
    df = prepare_corpus(fetch_arxiv(query=query, max_results=120))
    if df.empty:
        return None

    idx = VectorIndex(df["norm"].tolist())
    hits = idx.search(f"{query} {level} research", k=30)
    seed_df = df.iloc[[i for i, _ in hits]]

    return {
        "domain": domain_label,
        "level": level,
        "seed_df": seed_df,
        "topics": generate_topics(seed_df["norm"].tolist()),
        "ranked": rank_topics(seed_df, f"{query} {level} research", level),
    }


def feedback_callback(domain, level, title, vote):
    save_feedback(domain, level, title, vote)
    st.session_state["toast"] = "Thanks, feedback recorded!"


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🎓 SmartResearch Advisor")
st.write("AI-powered research topic generator for students (with ranking + feedback)")

domain_label = st.selectbox("Choose a domain", list(DOMAINS))
level = st.selectbox("Your level", LEVELS)

if st.button("Generate Topics"):
    with st.spinner("Fetching and processing data... please wait."):
        try:
            # Results live in session_state so feedback clicks (which rerun the
            # script) don't wipe the page.
            st.session_state["result"] = run_pipeline(domain_label, level)
        except Exception as e:  # network / arXiv errors
            st.session_state["result"] = None
            st.error(f"Could not fetch or process papers: {e}")

if "toast" in st.session_state:
    st.toast(st.session_state.pop("toast"))

result = st.session_state.get("result")
if result:
    st.subheader("✅ Suggested Topics")
    for i, row in result["ranked"].iterrows():
        st.markdown(f"### {row['title']}")
        st.write(row["abstract"][:300] + "...")
        st.markdown(f"[Read Paper]({row['url']})")

        col1, col2 = st.columns(2)
        col1.button("👍 Helpful", key=f"up{i}", on_click=feedback_callback,
                    args=(result["domain"], result["level"], row["title"], 1))
        col2.button("👎 Not helpful", key=f"down{i}", on_click=feedback_callback,
                    args=(result["domain"], result["level"], row["title"], -1))

    st.subheader("🔎 Topic Clusters (from BERTopic)")
    if result["topics"]:
        for t in result["topics"]:
            st.markdown(f"- {t}")
    else:
        st.caption("Not enough papers to form distinct clusters.")

    st.download_button(
        "📥 Download Papers (CSV)",
        data=result["seed_df"].to_csv(index=False),
        file_name="seed_papers.csv",
        mime="text/csv",
    )
