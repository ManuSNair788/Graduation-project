---
title: AJIO Discovery Engine
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.62.0
app_file: app.py
pinned: false
---

# AJIO Discovery Engine

**Live app:** [https://ajio-discovery-engine.streamlit.app/](https://ajio-discovery-engine.streamlit.app/)

A ranked, scored analysis of purchase barriers extracted from public user feedback about AJIO and
Myntra (Google Play reviews, YouTube comments) — not a summary or sentiment report.

## What it does

1. **Scrape** public reviews/comments into a raw corpus (`scraper.py`).
2. **Stage 1** — cheap-model relevance filter: is this snippet about saving, wishlisting,
   deferring, hesitating over, or abandoning a fashion purchase?
3. **Stage 2** — strict structured extraction per snippet (barrier, save intent, intensity,
   whether it's addressable without any monetary incentive, verbatim evidence).
4. **Stage 3** — pure-code aggregation and scoring:
   `Opportunity Score = frequency_percent × mean_intensity × addressability_weight`.
5. **Stage 4** — a stronger model synthesizes the top three opportunity areas from the aggregate
   table only (never the raw snippets).

The Streamlit app has four tabs: **Run** (load cached results instantly, or paste your own
reviews to run the pipeline live), **Opportunity table** (ranked barriers + chart + formula),
**Evidence** (verbatim snippets per barrier), and **Questions** (ten specific questions answered
from the data, each with its supporting number).

## Local development

```
pip install -r requirements.txt
streamlit run app.py
```

Scraping the corpus separately requires `requirements-scraper.txt` and is never run on the
deployed app — the corpus is pre-computed and committed under `data/`.

## Environment variables

`GROQ_API_KEY` (secret) plus `GROQ_MODEL_CHEAP`, `GROQ_MODEL_STRONG`, `GROQ_MODEL_SYNTHESIS`,
`GROQ_MODEL_FILTER` (not secret — just current model identifiers). See `.env.example`.
