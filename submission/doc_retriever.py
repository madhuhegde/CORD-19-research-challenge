import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi


class DocRetriever():
    """
    pip install git+ssh://git@github.com/dorianbrown/rank_bm25.git

    Parameters
    ----------
    lowercase : boolean
        Convert all characters to lowercase before tokenizing. (default is True)
   
    tokenizer : callable or None (default is None)
        Default uses " " to split sentences to words
    
    top_n : int (default 20)
        maximum number of top documents to retrieve
    
   
    """

    def __init__(self, lowercase=True, tokenizer=None, top_n=20):

        self.lowercase = lowercase
        self.tokenizer = tokenizer
        self.top_n = top_n
        self.vectorizer = BM25Okapi
        

    def fit_retriever(self, df):
        corpus = [document for document in df['paragraphs']]
        
        #by default no tokenizer. Use " " to spilt the sentences into words
        if(self.tokenizer == None):
           tokenized_corpus = [document.split(" ") for document in corpus]
        else:
           tokenizer = self.tokenizer
           tokenized_corpus = [tokenizer(document) for document in corpus]
           
        self.bm25 = self.vectorizer(tokenized_corpus)
  

    def compute_scores(self, query):
        if(self.tokenizer == None):
           tokenized_query = query.split(" ")
        else:
           tokenizer = self.tokenizer
           tokenized_query = tokenizer(query)
      
        doc_scores = self.bm25.get_scores(tokenized_query)

        #return top_n indices and scores as list
        sorted_scores = np.argsort(doc_scores)
        top_n = self.top_n
        out = zip(sorted_scores[-1:-top_n-1:-1],doc_scores[sorted_scores[-1:-top_n-1:-1]])
        return list(out)
