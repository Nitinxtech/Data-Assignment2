# Aurora Skies RAG Chatbot (One‑File)

A tiny Retrieval‑Augmented chatbot for Aurora Skies Airways FAQs. It’s intentionally simple and lives mostly in a single file (`app.py`).

- Retrieval: lightweight BM25‑style search with stemming, bigrams and a few hand‑written synonyms so paraphrases still match.
- Generation: prefers Groq (open‑source models hosted by Groq); optional local `transformers` for paraphrasing. If neither are available, it returns the best FAQ answer verbatim (grounded, no fluff).
- UI: Streamlit
- Dataset: `airline_faq.csv`

## Quickstart

```bash
# 1) Create an environment
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell

# 2) Install deps (minimal)
pip install -r requirements.txt


# 4) Run
streamlit run app.py
```