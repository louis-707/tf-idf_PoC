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


st.title("Patent similarity/ prior art search")
st.subheader("This is an example implementation of a specialized patent similarity/prior art search-tool based on TF-IDF-Vectorizer and Cosine Similarity.")
example = "A chromatic, confocal measuring device uses a broadband, high-intensity light source implemented by optically pumping a luminophore. The illumination of the luminophore is selected so that the properties of the luminophore are exploited to maximize the optical output power of the light source."

if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if st.button("Input Example US2024167808A1"):
    st.session_state.query_text = example
query = st.text_area("Enter a search query (e.g. patent abstract or description). \n" +
                     "Includes patents of CPC-Categories: G01b9/02036 , G01b9/02044 , G01b9/02042 , G01b2210/50. \n"+
                     "Measuring instruments characterised by the use of optical techniques \n"+
                     "by using chromatic effects, e.g. a wavelength dependent focal point, \n"+
                     "confocal imaging, \n"+
                     "and using chromatic effects to achieve wavelength-dependent depth resolution.",
    value=st.session_state.query_text,
    key="query_text")

if st.button("find prior art"):
    if query.strip():
        results = compute_query_similarity(query, vectorizer, tfidf_matrix, df['displayKey'], df['abstract'])
        st.subheader("top hits")
        st.write(results)
    else:
        st.warning("emtpy query")
