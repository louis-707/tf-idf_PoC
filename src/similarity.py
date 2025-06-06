from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from src.preprocessing import preprocess

def compute_similarity(tfidf_matrix, ids):


    cos_sim = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(cos_sim, index=ids, columns=ids)

def compute_query_similarity(query, vectorizer, tfidf_matrix, ids, processed):
    '''

    :param query: custom user query, will be compared to tfidf_matrix
    :param vectorizer: important to use same vectorizer for qery
    :param tfidf_matrix:
    :param ids:
    :param processed:
    :return: pd data-json frame
    '''
    cleaned_query = preprocess(query) # important to treat equally to patents used for similarity matrix
    query_vec = vectorizer.transform([cleaned_query])
    sim= cosine_similarity(query_vec, tfidf_matrix)
    similarities = sim.flatten()



    return pd.DataFrame({'id': ids,'similarity': similarities,'abstract': processed }).sort_values(by='similarity', ascending=False)
