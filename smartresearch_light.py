import arxiv
import pandas as pd
import spacy
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="SmartResearch Advisor (Light)", layout="centered")


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp, SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_papers(query, max_results=20):
    client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=3)
    search = arxiv.Search(query=query, max_results=max_results,
                          sort_by=arxiv.SortCriterion.SubmittedDate)
    return pd.DataFrame(
        [{"title": r.title, "summary": r.summary, "url": r.entry_id} for r in client.results(search)]
    )


def pick_diverse(embeddings, k=5):
    """Greedy farthest-point selection so the suggested topics differ from each other."""
    chosen = [0]
    sims = util.cos_sim(embeddings, embeddings)
    while len(chosen) < min(k, len(embeddings)):
        # candidate whose closest already-chosen paper is least similar
        best = min(
            (i for i in range(len(embeddings)) if i not in chosen),
            key=lambda i: max(float(sims[i][j]) for j in chosen),
        )
        chosen.append(best)
    return chosen


nlp, embedder = load_models()

st.title("🎓 SmartResearch Advisor (Light Version)")
st.write("Fast & lightweight topic generator for Streamlit Cloud")

domain = st.text_input("Enter your research domain (e.g., Machine Learning, Healthcare)")
level = st.selectbox("Select your level", ["Undergraduate", "Postgraduate", "PhD"])

if st.button("Generate Topics"):
    if not domain.strip():
        st.warning("Please enter a domain.")
    else:
        with st.spinner("Fetching research papers..."):
            try:
                df = fetch_papers(domain.strip())
            except Exception as e:
                st.error(f"Could not fetch papers: {e}")
                st.stop()

        if df.empty:
            st.error("No papers found. Try another domain.")
        else:
            st.success("Found some papers! Generating topics...")
            df = df.drop_duplicates("title").reset_index(drop=True)
            emb = embedder.encode(df["title"].tolist(), convert_to_tensor=True)

            st.subheader("🔑 Suggested Research Topics")
            for n, i in enumerate(pick_diverse(emb, k=5), 1):
                st.markdown(f"**{n}. {df.loc[i, 'title']}**  \n[Read paper]({df.loc[i, 'url']})")
