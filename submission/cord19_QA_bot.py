#!/usr/bin/env python -W ignore::DeprecationWarning
# coding: utf-8


import pandas as pd
import numpy as np
import os
#import torch

#from transformers import BertTokenizer, BertForQuestionAnswering
from doc_retriever import DocRetriever
from doc_reader import DocReader
from data_process import generate_data



if __name__ == "__main__":
  
  #check if corpus exists
  data_file = './covid_corpus.csv'
  isFile = os.path.isfile(data_file)
  if isFile == False:
     #Generating corpus file
     printf("Generating Corpus CSV File...")
     generate_data()

# Read corpus
  with open(data_file, 'rb') as f:
     frame = pd.read_csv(f)

#Document retriver based on BM250    
  retriever = DocRetriever(top_n=10)

#Vectorize BM250 using the corpus
  retriever.fit_retriever(frame)

#Reader based on Hugging Face BertForQuestionAnswering Transformer
#reader = DocReader('./model/')
  reader = DocReader('bert-large-uncased-whole-word-masking-finetuned-squad')

# Find top_n documents based on BM250 for the query 

  query = input("Enter the query (type exit to quit) : ")
  while(query != 'exit' and query != 'Exit'):
     print("Processing...")
     doc_scores = retriever.compute_scores(query)
   
#Select top_n documents
     index = [score[0] for score in doc_scores]
     text = frame.loc[index]

#predict n_best answers using BERT
     answers = reader.predict(df=text, query=query, n_best=5)

#Select best answer based on logitic values
     b_answer = reader.best_answer(answers)

     print('query: {}\n'.format(query))
     print('answer: {}\n'.format(b_answer['answer']))
     print('title: {}\n'.format(text['title'][b_answer['index']]))
     print('authors: {}\n'.format(text['authors'][b_answer['index']]))
     print('paragraph: {}\n'.format(b_answer['sentence']))
     
     query = input("Enter the next query (type exit to quit) : ")
 
  print("Done!")



