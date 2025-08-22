# physicsTrend
Explore 20-year research trends on arXiv across **physics** and **cond-mat** using precomputed BERTopic artifacts — **no large model required**. Browse topics, see top terms, read representative abstracts, and plot yearly prevalence.

> **Live demo:** https://physicstrend-gyc5hefqjfhjtymkheyvbb.streamlit.app/

> **Training pipeline (find the trend in you own area):** https://github.com/ChenglingL/abstract-trend

---

## Features

- **Fast, model-free UI** — runs only on CSV artifacts
- **Topic browser** with phrase-level labels
- **Concept search** (e.g., “glass transition”, “spin glass”, “amorphous solid”)
- **Representative documents** (shows **full abstracts** when available)
- **Yearly trend plots** (raw + 3-year smoothed share)

---

## Required files

The app expects these **CSV** files in `outputs/`:

- `topic_info.csv`  
  Columns: `Topic,int`, `Count,int` (+ optional `PrettyName`, `Name`, `Representation`)

- `topic_terms.csv`  
  Columns: `topic,int`, `term,str`, `weight,float`

- `topic_trends_by_year.csv`  
  Columns: `year,int`, `topic,int`, `count,int`, `n,int`, `share,float`

- `topic_rep_docs.csv` *(optional but recommended)*  
  Preferred columns:
  - `topic,int`, `rank,int`, `doc_full,str` (full abstract)  
  Optional: `title,str`, `doc_id,str`
  ---

## Quick start (local)

Requirements: **Python 3.11**

```bash
# Clone and enter the repo
git clone <your-app-repo-url>
cd <repo>

# Create & activate a virtual env
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# Install Lite dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py