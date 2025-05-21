#from src.search import find_top_k_similar
from src.preprocessing import clean_text
from src.similarity import *
from src.vectorizer import build_vectorizer
import pandas as pd


def main():

 clean_text('data/raw.csv') # writes processed date in processed.csv
 df = pd.read_csv('data/processed.csv')
 # df['processed'] = preprocessed abstract

 vectorizer, tfidf_matrix = build_vectorizer(df['processed'])
 print(f"\n only tfidf-matrix of csv patents:\n", compute_similarity(tfidf_matrix, df['id']))

 query = "a scanning optical devices which works by splitting  light into different colors"
 #query = "optical device which works by splitting light in two separate beams and analyzing the optical path difference"


 print("\n top similar patents:")
 print(compute_query_similarity(query, vectorizer, tfidf_matrix, df['id'], df['processed']))

if __name__ == "__main__":
 main()
