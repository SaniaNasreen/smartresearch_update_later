# 🎓 SmartResearch Advisor

SmartResearch Advisor is an **AI-powered tool** that helps students generate **personalized research paper topics** using NLP and recent academic papers from arXiv.

---

## 🚀 Features
- Domain + Level input → personalized topic suggestions
- Fetches recent research papers (arXiv)
- NLP preprocessing (cleaning, lemmatization, stopwords removal)
- Embeddings + FAISS search for relevance
- BERTopic clustering for trending areas
- Ranking based on recency × diversity (MMR) × difficulty (matched to your level)
- Streamlit UI with feedback buttons (👍/👎) stored in SQLite
- CSV export of seed papers

Two apps are included:

| File | Purpose |
|------|---------|
| `smartresearch_app.py` | Full app (FAISS + BERTopic + ranking + feedback) |
| `smartresearch_light.py` | Lightweight version for low-memory hosting (e.g. Streamlit Cloud) |

---

## ⚙️ Installation

1. Clone this repo:
```bash
git clone https://github.com/SaniaNasreen/smartresearch_update_later.git
cd smartresearch_update_later
```

2. Create a virtual environment and install dependencies (Python 3.11 recommended):
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run smartresearch_app.py
# or the light version
streamlit run smartresearch_light.py
```

The spaCy model and NLTK stopwords are downloaded automatically on first run.

---

## 🐳 Docker

```bash
docker build -t smartresearch .
docker run -p 8501:8501 smartresearch
```

Then open http://localhost:8501.

---

## 🗂️ Project structure

```
smartresearch_app.py      # full Streamlit app
smartresearch_light.py    # lightweight app
requirements.txt          # Python dependencies
packages.txt              # system packages (Streamlit Cloud / Codespaces)
Dockerfile                # container build
.devcontainer/            # GitHub Codespaces config
```

---

## 📝 Notes
- Feedback is saved locally to `feedback.db` (git-ignored).
- arXiv requests are cached for one hour to avoid hitting rate limits.
