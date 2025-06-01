from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import scipy.sparse
import os

def save_vectorizer_and_matrix(vectorizer, tfidf_matrix, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    scipy.sparse.save_npz(os.path.join(model_dir, 'tfidf_matrix.npz'), tfidf_matrix)
    print("[Info] model + matrix saved")

def load_vectorizer_and_matrix(model_dir='models'):
    vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    tfidf_matrix = scipy.sparse.load_npz(os.path.join(model_dir, 'tfidf_matrix.npz'))
    return vectorizer, tfidf_matrix

def build_vectorizer(corpus):
    '''
    :param corpus:
    :return:
    '''
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

