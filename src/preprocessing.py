import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) # https://www.geeksforgeeks.org/removing-stop-words-nltk-python/



def clean_text(file):
    """
    -> load csv from /data/raw.csv
       write csv to /data/processed.csv
    :param file: path
    :return: boolean
    """
    try:
        df = pd.read_csv(file)

        df['processed'] = df['abstract'].apply(preprocess) # for the first tests only the abstract is processed,
        # later also the whole description and Claims, also back/ forward citations can be useful to determine scope and breadth of invention
        #https://ftp.zew.de/pub/zew-docs/div/innokonf/6bsampatziedonis.pdf

        output_path = os.path.join(os.path.dirname(file), 'processed.csv')
        df.to_csv(output_path, index=False)

        print( {output_path})
        return True

    except Exception as e:
        print({e})
        return False


def preprocess(text: str) -> str:
    text = text.lower()
    return ' '.join([
        word for word in text.split()
        if word not in stop_words and word not in string.punctuation
    ])