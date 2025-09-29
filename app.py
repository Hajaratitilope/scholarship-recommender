# app.py
# Requirements: pandas, openpyxl, scikit-learn, python-dateutil, streamlit

import streamlit as st
import pandas as pd
import numpy as np
from dateutil import parser as dateparser
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="Scholarship Recommender (Prototype)", layout="wide")

# ------------------- Helpers -------------------
def robust_rename_columns(df):
    expected = {
        'scholarship name': 'scholarship_name',
        'scholarship': 'scholarship_name',
        'donor': 'host',
        'host': 'host',
        'type of funding': 'funding_type',
        'funding': 'funding_type',
        'eligibility': 'eligibility',
        'benefits': 'benefits',
        'application deadline': 'deadline',
        'deadline': 'deadline',
        'official source link': 'source_link',
        'link': 'source_link'
    }
    rename_map = {}
    for orig, target in expected.items():
        for c in df.columns:
            if orig in c.lower() and c not in rename_map:
                rename_map[c] = target
    return df.rename(columns=rename_map)

def parse_deadline_cell(s):
    if pd.isna(s) or str(s).strip() == "":
        return (pd.NaT, 'missing')
    s_low = str(s).lower()
    if any(x in s_low for x in ['rolling','open','ongoing','continuous','no deadline','contact']):
        return (pd.NaT, 'rolling')
    if any(x in s_low for x in ['tba','to be announced','tbd','n/a','not specified']):
        return (pd.NaT, 'unknown')
    try:
        dt = dateparser.parse(str(s), fuzzy=True)
        return (pd.to_datetime(dt).normalize(), 'date')
    except Exception:
        return (pd.NaT, 'parse_error')

# ------------------- Data loader -------------------
@st.cache_data(ttl=3600)
def load_data(local_path="scholarships.xlsx", uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        if not os.path.exists(local_path):
            st.error(f"Local file '{local_path}' not found!")
            return pd.DataFrame()
        df = pd.read_excel(local_path, engine="openpyxl")

    df = robust_rename_columns(df)

    for col in ['scholarship_name','host','funding_type','eligibility','benefits','deadline','source_link']:
        if col not in df.columns:
            df[col] = ""

    # clean whitespace
    for c in ['scholarship_name','host','funding_type','eligibility','benefits','source_link']:
        df[c] = df[c].astype(str).str.strip().replace('nan','')

    # parse deadlines
    parsed = df['deadline'].apply(parse_deadline_cell)
    df['deadline_parsed'] = parsed.apply(lambda x: x[0])
    df['deadline_type'] = parsed.apply(lambda x: x[1])
    today = pd.Timestamp.now().normalize()
    df['is_expired'] = df['deadline_parsed'].notna() & (df['deadline_parsed'] < today)

    # dedupe key
    df['dup_key'] = df['scholarship_name'].str.lower().str.strip() + " | " + df['host'].str.lower().str.strip()
    df['is_duplicate'] = df.duplicated('dup_key', keep='first')

    # create text doc for similarity
    df['doc'] = (
        df['scholarship_name'].fillna('') + ' ' +
        df['eligibility'].fillna('') + ' ' +
        df['benefits'].fillna('')
    ).str.replace(r'\s+',' ', regex=True).str.lower().fillna('')

    return df

@st.cache_data(ttl=3600)
def build_vectorizer_and_matrix(docs_series):
    if docs_series.isna().all() or (docs_series.str.strip() == "").all():
        return None, None
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    mat = vectorizer.fit_transform(docs_series.fillna(''))
    return vectorizer, mat

# ------------------- Recommender -------------------
def recommend(df, profile, vectorizer, doc_matrix, top_n=10, alpha=0.65):
    df_local = df.copy()
    prof_text = " ".join([str(v) for v in profile.values() if v]).lower()
    cosine_sims = np.zeros(len(df_local))
    if vectorizer is not None and doc_matrix is not None:
        prof_vec = vectorizer.transform([prof_text])
        cosine_sims = cosine_similarity(prof_vec, doc_matrix).flatten()
    df_local['final_score'] = cosine_sims
    return df_local.sort_values('final_score', ascending=False).head(top_n)

# ------------------- UI -------------------
st.title("Scholarship Recommender: Prototype 🎓")
st.write("Upload your Excel dataset or use the local `scholarships.xlsx` file.")

uploaded = st.file_uploader("Upload an Excel file (.xlsx)", type=['xlsx'])
use_local = st.checkbox("Use local scholarships.xlsx", value=True)
top_n = st.number_input("Number of results", min_value=3, max_value=50, value=10, step=1)

if uploaded:
    df = load_data(uploaded_file=uploaded)
elif use_local:
    df = load_data()
else:
    df = pd.DataFrame()
    st.info("Upload a file or check 'Use local scholarships.xlsx'.")

vectorizer, doc_matrix = build_vectorizer_and_matrix(df['doc']) if not df.empty else (None, None)

if st.button("Run recommendation") and not df.empty:
    profile = {"level": "", "country": "", "field": ""}
    results = recommend(df, profile, vectorizer, doc_matrix, top_n=top_n)
    st.dataframe(results)
