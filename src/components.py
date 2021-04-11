import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import requests
import json
import seaborn as sns
import re
#matplotlib inline

## Importing Textblob package
from textblob import TextBlob
from textblob import Word

# Importing CountVectorizer for sparse matrix/ngrams frequencies
from sklearn.feature_extraction.text import CountVectorizer

## Import datetime
import datetime as dt

import nltk
nltk.download('stopwords')
import itertools
import chardet
from nltk.corpus import stopwords
from difflib import SequenceMatcher



import spacy
from rank_bm25 import BM25Okapi
from tqdm import tqdm
STOPWORDS = stopwords.words('english')
STOPWORDS = set(STOPWORDS)

def text_prepare(text, STOPWORDS):
    """
        text: a string
        
        return: a clean string
    """
    REPLACE_BY_SPACE_RE = re.compile('[\n\"\'/(){}\[\]\|@,;#]')
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.lower()

    text = ' '.join([word for word in text.split() if word not in STOPWORDS]) 
    text = text.strip()
    return text


class TextProcess:
    def search_text(self, query):
        df=pd.read_csv('issues.csv', delimiter=',')
        df["SUBJECT"] = df["Subject"].apply(lambda x:text_prepare(x,STOPWORDS) )
        df["SUBJECT"] = df["SUBJECT"].apply(lambda x: " ".join([Word(myword).lemmatize() for myword in x.split()])  )
        df["Result"] = df.SUBJECT +df.Author
        nlp = spacy.load("en_core_web_sm")
        text_list = df.SUBJECT.str.lower().values
        tok_text=[] # for our tokenised corpus
        #Tokenising using SpaCy:
        for doc in tqdm(nlp.pipe(text_list, disable=["tagger", "parser","ner"])):
          tok = [t.text for t in doc if t.is_alpha]
        tok_text.append(tok)
        
        tokenized_corpus = [query.lower().split(" ") for query in df.Result.values]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        results = bm25.get_top_n(tokenized_query,df.Result.values,n=3)
        return  results

