import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
import os
import json
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) # https://www.geeksforgeeks.org/removing-stop-words-nltk-python/



def preprocess_csv(file):
    """
    -> load csv from /data-json/raw.csv
       write csv to /data-json/processed.csv
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

        print(f"Processed csv saved: {output_path}")
        return True

    except Exception as e:
        print({e})
        return False
def preprocess_json(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            raw = f.read()

        # lens.org exports only as a single json object
        if not raw.strip().startswith('['):
            raw = f"[{raw}]"

        data = json.loads(raw)

        for it in data:
            if 'abstract' in it:
                it['processed'] = preprocess(it['abstract'])

        output_path = os.path.join(os.path.dirname(file), 'processed.json')
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=2, ensure_ascii=False)

        print(f"Processed json saved: {output_path}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

def preprocess(text: str) -> str:
    text = text.lower()
    return ' '.join([
        word for word in text.split()
        if word not in stop_words and word not in string.punctuation
    ])