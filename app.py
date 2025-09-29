# app.py
# Requirements: pandas, openpyxl, scikit-learn, python-dateutil, streamlit

import streamlit as st
import pandas as pd
import numpy as np
from dateutil import parser as dateparser
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="Scholarship Recommender (Prototype)", layout="wide")

# ------------------- Helpers -------------------
def robust_rename_columns(df):
    # map common names to canonical keys using substring matching
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
    cols_lower = {c: c.lower() for c in df.columns}
    rename_map = {}
    for orig, target in expected.items():
        for c in df.columns:
            if orig in c.lower() and c not in rename_map:
                rename_map[c] = target
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def parse_deadline_cell(s):
    if pd.isna(s) or str(s).strip()=="":
        return (pd.NaT, 'missing')
    s_str = str(s).strip()
    s_low = s_str.lower()
    # rolling / ongoing signals
    if any(x in s_low for x in ['rolling','open','ongoing','continuous','no deadline','contact']):
        return (pd.NaT, 'rolling')
    if any(x in s_low for x in ['tba','to be announced','tbd','n/a','not specified']):
        return (pd.NaT, 'unknown')
    # try parsing date
    try:
        dt = dateparser.parse(s_str, fuzzy=True, dayfirst=False)
        return (pd.to_datetime(dt).normalize(), 'date')
    except Exception:
        return (pd.NaT, 'parse_error')

# ------------------- Data loader -------------------
@st.cache_data(ttl=3600)
def load_data_from_excel(local_path=None, uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            raise
    else:
        if local_path is None:
            raise ValueError("No file provided")
        df = pd.read_excel(local_path, engine="openpyxl")
    df = df.copy()
    df = robust_rename_columns(df)
    # ensure columns exist
    for col in ['scholarship_name','host','funding_type','eligibility','benefits','deadline','source_link']:
        if col not in df.columns:
            df[col] = ""
    # clean whitespace
    text_cols = ['scholarship_name','host','funding_type','eligibility','benefits','source_link']
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip().replace('nan','')
    # parse deadlines
    parsed = df['deadline'].apply(parse_deadline_cell)
    df['deadline_parsed'] = parsed.apply(lambda x: x[0])
    df['deadline_type'] = parsed.apply(lambda x: x[1])
    today = pd.Timestamp.now().normalize()
    df['is_expired'] = df['deadline_parsed'].notna() & (df['deadline_parsed'] < today)
    # dedupe key
    def norm_text(x):
        return str(x).lower().strip()
    df['dup_key'] = df['scholarship_name'].apply(norm_text) + " | " + df['host'].apply(norm_text)
    df['is_duplicate'] = df.duplicated('dup_key', keep='first')
    # doc for text matching
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
    # rule-based scoring
    def level_match(row):
        lvl = profile.get('level','') or ''
        if not lvl: return 0
        return 1 if lvl.lower() in str(row['eligibility']).lower() else 0
    def field_match(row):
        field = profile.get('field','') or ''
        if not field: return 0
        tokens = field.lower().split()
        txt = str(row['doc']).lower()
        return 1 if any(tok in txt for tok in tokens) else 0
    def country_match(row):
        country = profile.get('country','') or ''
        if not country: return 0
        txt = str(row['doc']).lower() + " " + str(row['eligibility']).lower()
        if country.lower() in txt: return 1
        # quick Africa fallback
        if 'africa' in txt and country.lower() in ['nigeria','ghana','kenya','south africa','uganda','tanzania','rwanda','ethiopia']:
            return 1
        if 'international' in txt or 'worldwide' in txt or 'global' in txt:
            return 1
        return 0
    df_local['level_match'] = df_local.apply(level_match, axis=1)
    df_local['field_match'] = df_local.apply(field_match, axis=1)
    df_local['country_match'] = df_local.apply(country_match, axis=1)
    # raw combined rule
    df_local['rule_raw'] = df_local['level_match']*0.5 + df_local['field_match']*0.3 + df_local['country_match']*0.2
    if df_local['rule_raw'].max() > 0:
        df_local['rule_score'] = df_local['rule_raw'] / df_local['rule_raw'].max()
    else:
        df_local['rule_score'] = 0.0
    final = alpha * cosine_sims + (1-alpha) * df_local['rule_score'].values
    # penalize expired
    final = final - (df_local['is_expired'].astype(int) * 0.2)
    final = np.clip(final, 0, None)
    df_local['final_score'] = final
    # sort and return top_n
    res = df_local.sort_values('final_score', ascending=False).head(top_n)
    # friendly columns
    res = res[['scholarship_name','host','funding_type','deadline','deadline_parsed','is_expired','final_score','source_link']]
    return res

# ------------------- UI -------------------
st.title("Scholarship Recommender: The Prototype 🎓")
st.write("Prototype demo: recommend scholarships from a dataset (Excel).")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Load data")
    uploaded = st.file_uploader("Upload an Excel file (.xlsx) — or leave empty to use local 'scholarships.xlsx'", type=['xlsx'])
    use_local = st.checkbox("Use local scholarships.xlsx (file in folder)", value=True)
    if not uploaded and not use_local:
        st.info("Upload an Excel file or check 'Use local scholarships.xlsx'.")
    # alpha slider
    alpha = st.slider("Text similarity weight (alpha)", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    top_n = st.number_input("Number of results", min_value=3, max_value=50, value=10, step=1)

with col2:
    st.header("Student profile (sample)")
    level = st.selectbox("Level of study", ["", "Undergraduate", "Masters", "PhD", "Highschool"])
    country = st.text_input("Country", value="Nigeria")
    field = st.text_input("Field of study", value="Computer Science")

# load data
try:
    if uploaded:
        df = load_data_from_excel(uploaded_file=uploaded)
    elif use_local:
        if os.path.exists("scholarships.xlsx"):
            df = load_data_from_excel(local_path="scholarships.xlsx")
        else:
            df = pd.read_csv("data/sample_scholarships.csv")
    else:
        df = pd.DataFrame()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

st.write("")  # spacer

# build vectorizer
vectorizer, doc_matrix = build_vectorizer_and_matrix(df['doc'])

# show diagnostics
with st.expander("Dataset Overview (click to expand)"):
    st.write(f"Total rows: {len(df)}")
    st.write(f"Expired: {int(df['is_expired'].sum())}")
    st.write(f"Rolling/Ongoing: {int((df['deadline_type']=='rolling').sum())}")
    st.write(f"Deadline parse issues (parse_error/unknown/missing): {int(df['deadline_type'].isin(['parse_error','unknown','missing']).sum())}")
    st.write(f"Duplicates (simple name+host): {int(df['is_duplicate'].sum())}")
    st.write("Sample rows (first 5):")
    st.dataframe(df.head(5), height=240)

if st.button("Run recommendation"):
    with st.spinner("Scoring scholarships..."):
        profile = {"level": level, "country": country, "field": field}
        results = recommend(df, profile, vectorizer, doc_matrix, top_n=top_n, alpha=alpha)
        if results.empty:
            st.warning("No results found (dataset empty or text fields too sparse).")
        else:
            st.success(f"Top {len(results)} recommendations")
            for idx, row in results.iterrows():
                st.markdown(f"### {row['scholarship_name']} — {row['host']}")
                st.markdown(f"- Funding: {row['funding_type']}")
                dl = row['deadline_parsed'] if pd.notna(row['deadline_parsed']) else row['deadline']
                st.markdown(f"- Deadline: {dl} {'(expired)' if row['is_expired'] else ''}")
                if pd.notna(row['source_link']) and str(row['source_link']).strip() != '':
                    st.markdown(f"- [Source]({row['source_link']})")
                st.markdown(f"- Score: {row['final_score']:.4f}")
                st.markdown("---")
            # make a CSV for download
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download results (CSV)", data=csv, file_name="recommendations.csv", mime="text/csv")