import streamlit as st

import pandas as pd

from src.preprocessing import preprocess_json
from src.similarity import compute_query_similarity
from src.vectorizer import build_vectorizer

preprocess_json('data-json/raw.json')
df = pd.read_json('data-json/processed.json')
df = df.dropna(subset=['processed'])
df = df[df['processed'].str.strip().astype(bool)]
print(df[['abstract', 'processed']].head())
vectorizer, tfidf_matrix = build_vectorizer(df['processed'])


st.title("Patent similarity search")

query = st.text_area("Enter a search query (e.g. patent abstract or description")

if st.button("find simmilar patents"):
    if query.strip():
        results = compute_query_similarity(query, vectorizer, tfidf_matrix, df['displayKey'], df['abstract'])
        st.subheader("top hits")
        st.write(results)
    else:
        st.warning("emtpy query")