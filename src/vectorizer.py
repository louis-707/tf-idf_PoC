from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(corpus):
    '''

    :param corpus:
    :return:
    '''
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix