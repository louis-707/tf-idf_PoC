import pandas as pd
from src.preprocessing import preprocess



def load_and_prepare(path):
    df = pd.read_csv(path)
    df['processed'] = df['abstract'].apply(preprocess) ##
    return df
