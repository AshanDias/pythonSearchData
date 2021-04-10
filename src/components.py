import concurrent.futures
import itertools
import operator
import re
import csv

import requests
import wikipedia
from gensim.summarization.bm25 import BM25
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline
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

    # delete stopwords from text
    text = ' '.join([word for word in text.split() if word not in STOPWORDS]) 
    text = text.strip()
    #text = text.apply(lambda x: " ".join([Word(myword).lemmatize() for myword in x.split()])  )
    return text

class QueryProcessor:

    def __init__(self, nlp, keep=None):
        self.nlp = nlp
        self.keep = keep or {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'}

    def generate_query(self, text):
        doc = self.nlp(text)
        query = ' '.join(token.text for token in doc if token.pos_ in self.keep)
        return query

class DataRetrieval:

    def search(self, query):

        i = 0
        result = []

        with open('issues.csv') as f_obj:
            reader = csv.reader(f_obj, delimiter=',')
            for line in reader:
                i += 1
                if i == 1:
                    continue
                if query in line[6]:
                    result.append(line)
        return result

class TextProcess:
    def search_text(self, query):
        result = []
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
        
        # bm25 = BM25Okapi(tok_text)
        
        tokenized_corpus = [query.lower().split(" ") for query in df.Result.values]
        bm25 = BM25Okapi(tokenized_corpus)
        import time
        t0 = time.time()
        tokenized_query = query.split(" ")
        results = bm25.get_top_n(tokenized_query,df.Result.values,n=3)
        # for i in results:
        #      result.append(i)
        return  results

class DocumentRetrieval:

    def __init__(self, url='https://en.wikipedia.org/w/api.php'):
        self.url = url

    def search_pages(self, query):
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json'
        }
        res = requests.get(self.url, params=params)
        return res.json()

    def search_page(self, page_id):
        res = wikipedia.page(pageid=page_id)
        return res.content

    def search(self, query):
        pages = self.search_pages(query)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            process_list = [executor.submit(self.search_page, page['pageid']) for page in pages['query']['search']]
            docs = [self.post_process(p.result()) for p in process_list]
        return docs

    def post_process(self, doc):
        pattern = '|'.join([
            '== References ==',
            '== Further reading ==',
            '== External links',
            '== See also ==',
            '== Sources ==',
            '== Notes ==',
            '== Further references ==',
            '== Footnotes ==',
            '=== Notes ===',
            '=== Sources ===',
            '=== Citations ===',
        ])
        p = re.compile(pattern)
        indices = [m.start() for m in p.finditer(doc)]
        min_idx = min(*indices, len(doc))
        return doc[:min_idx]


class PassageRetrieval:

    def __init__(self, nlp):
        self.tokenize = lambda text: [token.lemma_ for token in nlp(text)]
        self.bm25 = None
        self.passages = None

    def preprocess(self, doc):
        passages = [p for p in doc.split('\n') if p and not p.startswith('=')]
        return passages

    def fit(self, docs):
        passages = list(itertools.chain(*map(self.preprocess, docs)))
        corpus = [self.tokenize(p) for p in passages]
        self.bm25 = BM25(corpus)
        self.passages = passages

    def most_similar(self, question, topn=10):
        tokens = self.tokenize(question)
        scores = self.bm25.get_scores(tokens)
        pairs = [(s, i) for i, s in enumerate(scores)]
        pairs.sort(reverse=True)
        passages = [self.passages[i] for _, i in pairs[:topn]]
        return passages


class AnswerExtractor:

    def __init__(self, tokenizer, model):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

    def extract(self, question, passages):
        answers = []
        for passage in passages:
            try:
                answer = self.nlp(question=question, context=passage)
                answer['text'] = passage
                answers.append(answer)
            except KeyError:
                pass
        answers.sort(key=operator.itemgetter('score'), reverse=True)
        return answers
