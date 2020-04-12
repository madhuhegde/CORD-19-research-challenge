import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from nltk import tokenize

class DocReader():
    """
    Uses Hungging Face BertForQuestionAnswering

    Parameters
    ----------
    model     : path to folder containing pytorch.bin, bert_config.json and vocab.txt
                or pretrained model
    lowercase : boolean
        Convert all characters to lowercase before tokenizing. (default is True)
   
    tokenizer : default is BertTokenizer
       
    """

    def __init__(self, model:str=None, lowercase=True, tokenizer=BertTokenizer):

        self.lowercase = lowercase
        self.tokenizer = tokenizer.from_pretrained(model)
        self.model = BertForQuestionAnswering.from_pretrained(model)
        

    def predict(self, 
                df: pd.DataFrame = None,
                query: str = None,
                n_best: int =3):
    
        doc_text = df['paragraphs']
        self.n_best = n_best
        
        if(self.lowercase):
           query = query.lower()
        
        # num docs_index must be equal to top_n
        doc_index = list(doc_text.index)
        
        answers = []
        for df_index in doc_index:
        
           if(self.lowercase):
              doc_lines = doc_text[df_index].lower()
           else:
              doc_lines = doc_text[df_index]
                 
           doc_lines = tokenize.sent_tokenize(doc_lines)
           
           doc_answers = []
           for lines in doc_lines:
              input_ids = self.tokenizer.encode(query, lines)
              #print(lines, input_ids)
              token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
              start_scores, end_scores = self.model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

              all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
              answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
              
              #it is better to decode the tokens at later stage
              #answer = all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
              #answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer))
              entry = {}
              #print(answer)
              start_scores = start_scores.detach().numpy()
              end_scores = end_scores.detach().numpy()
              entry['answer'] = answer
              entry['score'] = (max(start_scores[0]), max(end_scores[0]))
              entry['index'] = df_index
              entry['sentence'] = lines
              doc_answers.append(entry)
              
           best_doc_ans = [entry['score'][0]+entry['score'][1]  for entry in doc_answers]
           ans_index = np.argsort(best_doc_ans)
           
           # take n_best answers per document based on max(start_scores+end_scores)
           #it is possible to improve by taking different metric
           for ans in range(1,self.n_best+1):
              answers.append(doc_answers[ans_index[-ans]])
           
              
        best_ans = [entry['score'][0]+entry['score'][1]  for entry in answers]
        ans_index = np.argsort(best_ans)     
        
        n_best_answers = []
        for ans in range(1,self.n_best+1):
           n_best_answers.append(answers[ans_index[-ans]])
        
        return(n_best_answers)
        
        
        
    def best_answer(self, answers):
        ans_dict = {}
        final_answer = {}
        ANS_THRESH = 2.0
        max_score = answers[0]['score'][0]+answers[0]['score'][1]
    
        for ans in answers:
           score = ans['score'][0]+ans['score'][1]
           if score > max_score - ANS_THRESH:
              start_end = ans['answer'].split()
              if(len(start_end)>0):
                 ans_key = (start_end[0], start_end[-1])
                 ans_dict[ans_key] = ans_dict.get(ans_key,0)+1
                
        inverse = [value for key, value in ans_dict.items()]
        inverse.sort()
        ans_list = [key for key, value in ans_dict.items() if(value == inverse[-1])]
    
        max_score = float('-inf')
        for ans in answers:
           start_end = ans['answer'].split()
           ans_key = (start_end[0], start_end[-1])
           for item in ans_list:
              if(ans_key == item):
                 score = ans['score'][0]+ans['score'][1]
                 if(score > max_score):
                    ans_ids = self.tokenizer.convert_tokens_to_ids(start_end)
                    final_answer['answer'] = self.tokenizer.decode(ans_ids)
                    final_answer['index'] = ans['index']
                    final_answer['sentence'] = ans['sentence']
        
        
        return(final_answer)
