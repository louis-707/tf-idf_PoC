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

query = st.text_area("Enter a search query (e.g. patent abstract or description). \n" +
                     "Includes patents of CPC-Categories: G01b9/02036 , G01b9/02044 , G01b9/02042 , G01b2210/50. \n"+
                     "Measuring instruments characterised by the use of optical techniques \n"+
                     "by using chromatic effects, e.g. a wavelength dependent focal point, \n"+
                     "confocal imaging, \n"+
                     "and using chromatic effects to achieve wavelength-dependent depth resolution.")

if st.button("find simmilar patents"):
    if query.strip():
        results = compute_query_similarity(query, vectorizer, tfidf_matrix, df['displayKey'], df['abstract'])
        st.subheader("top hits")
        st.write(results)
    else:
        st.warning("emtpy query")
if st.button("Try Example JP 2021001914"):

    example="To provide a confocal displacement meter capable of easily and accurately measuring a displacement in a measurement object.SOLUTION: Light having chromatic aberration is converged by a lens unit 220 and irradiated to a measurement object S from a measurement head 200. Light of wavelength reflected while focused on the surface of the measurement object S passes through an optical fiber 314 in the measurement head 200. The light passing through the optical fiber 314 is guided to a spectroscopic part 130 in a processing device 100 to be dispersed. In the processing device 100, the light dispersed by the spectroscopic part 130 is received by a light reception part 140, and a light reception signal outputted from the light reception part 140 is acquired by a control part 152. The control part 152 measures displacement on the basis of the acquired light reception signal, and gives the light reception signal to an external PC 600. A CPU 601 of a PC 600 causes a display part 700 to display change to a light reception signal acquired at the present time from the light reception signal acquired at a previous time before the current time as change information.SELECTED DRAWING: Figure 1"
    results = compute_query_similarity(example, vectorizer, tfidf_matrix, df['displayKey'], df['abstract'])
    st.subheader("top similar patents")
    st.write(results)