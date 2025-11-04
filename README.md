# STEM Analysis + ARMA Lab

This project builds a simple pipeline to:

- tokenize and **stem** text (Porter stemmer)
- aggregate stem frequencies by date
- fit an **ARMA(p,q)** model (via statsmodels) on a chosen stem's time series
- view and experiment in a small **Streamlit** dashboard

## Quickstart
```bash
python -m venv .venv
# Windows:
. .\.venv\Scripts\activate
pip install -r requirements.txt
# Dashboard:
streamlit run ui/App.py
# CLI example:
python -m src.main --csv data/sample_text.csv --stem traffic --p 1 --q 1 --steps 7
```
Sample input CSV: `data/sample_text.csv` with columns: `date,text`.
