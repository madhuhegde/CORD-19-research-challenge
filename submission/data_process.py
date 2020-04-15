#!/usr/bin/env python
# coding: utf-8

# Re-using some functions from https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv

import os
import re
import json
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from nltk import tokenize #import sent_tokenize


def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body



def format_body_text(body_text):
    
    body = ""

    for di in body_text:
        text = di['text']
        body += text
    return body
    
    
def format_corpus_text(body_text, min_len=18, max_len=128):
    junk_text = "copyright"
    
    def remove_braces_brackets(body_text):
        body_text = re.sub(r'\([0-9]+\)', '', body_text)
        body_text = re.sub(r'\[[^)]*\]', '', body_text)
        return(body_text)
        
    body_text = remove_braces_brackets(body_text)
    text_lines = []
    token_lines = tokenize.sent_tokenize(body_text)
    for line in token_lines:
      
        words = line.split()
        if junk_text not in words:
             max_word_len = len(max(words, key=len))
             if (len(words) > min_len) and (len(words) < max_len) and max_word_len > 5:
                 text_lines.append(line)
    
    return(text_lines)

def find_filenames(folder):
    filenames = os.listdir(folder)
    print("Number of articles retrieved from the folder:", len(filenames))
    files = []

    for filename in filenames:
        filename = folder + filename
        file = json.load(open(filename, 'rb'))
        files.append(file)
    return(files)    

def generate_csv_corpus(corpus_files, csv_file:str = None):
    out_file = []
  
    for file in tqdm(corpus_files):
       body_text = format_body_text(file['body_text'])
       body_text = body_text.replace('\n',' ')
   
       text = []
       text.append(body_text)
       features = [
          file['metadata']['title'],
          format_authors(file['metadata']['authors'], with_affiliation=True),
          text]
          
       out_file.append(features)
      
    col_names = [
        'title',
        'authors',
        'paragraphs']

    csv_df = pd.DataFrame(out_file, columns=col_names)

# CSV file is used by DocRetriver() and DocReader()
    csv_df.to_csv(csv_file, index=False)    
    return   
 
def generate_text_corpus(corpus_files, text_file:str=None):
    
# The covid_corpus.txt is raw text file used for pre-training of BERT
    corpus_text = []

    for file in tqdm(corpus_files):
      file_text = format_body_text(file['body_text'])
      file_lines = format_corpus_text(file_text)
      if(len(file_lines)>5):
         corpus_text.append(file_lines)

      with open(text_file, 'w') as corp_file:
         for lines in corpus_text:
            for line in lines:
                line = line.lower()
                corp_file.write("%s\n" %line)
            corp_file.write("\n")
        

def generate_data(dataset_folder=None, pre_training=False):
    #find all the files
    folder = dataset_folder+'/biorxiv_medrxiv/biorxiv_medrxiv/'
    all_files = find_filenames(folder)
    folder = biorxiv_dir = dataset_folder+'comm_use_subset/comm_use_subset/'
    all_files.extend(find_filenames(folder))
    folder = biorxiv_dir = dataset_folder+'/noncomm_use_subset/noncomm_use_subset/'
    all_files.extend(find_filenames(folder))
    folder = dataset_folder+'/custom_license/custom_license/'
    all_files.extend(find_filenames(folder))

    print("Total number of articles retrieved:", len(all_files))
    
    #generate csv corpus for BERT reader
    generate_csv_corpus(all_files, csv_file = './covid_corpus.csv')
    
    #generate text corpus for BERT pre-training (optional)  
    if(pre_training):
       generate_text_corpus(all_files, text_file = './covid_corpus.txt')
    
        
if __name__ == "__main__": 
    generate_data()