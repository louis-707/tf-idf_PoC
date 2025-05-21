from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from src.preprocessing import preprocess

def compute_similarity(tfidf_matrix, ids):


    cos_sim = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(cos_sim, index=ids, columns=ids)

def compute_query_similarity(query, vectorizer, tfidf_matrix, ids, processed):
    '''

    :param query: custom user querry, will be compared to tfidf_matrix
    :param vectorizer: important to use same vectorizer for qerry
    :param tfidf_matrix:
    :param ids:
    :param processed:
    :return: pd data frame
    '''
    cleaned_query = preprocess(query) # important to treat equally to patents used for similarity matrix
    query_vec = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()  # flatten returns a copy of the array collapsed into one dimension (numpy function) for better organized data frame



    return pd.DataFrame({'id': ids,'similarity': similarities,'text': processed }).sort_values(by='similarity', ascending=False)
