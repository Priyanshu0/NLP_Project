#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Priyanshu
"""

import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import pandas as pd
from textblob import TextBlob
from scipy import spatial
from torch import nn
import spacy
import nltk
import torch
nltk.download('punkt')
en_nlp = spacy.load('en')
train = pd.read_json("/Users/aditya/Downloads/train.json")
valid = pd.read_json("/Users/aditya/Downloads/dev-v2.0.json")

contexts = []
questions = []
answers_text = []
answers_start = []
for i in range(train.shape[0]):
    topic = train.iloc[i,1]['paragraphs']
    for sub_para in topic:
        for q_a in sub_para['qas']:
            if(q_a['answers']!=[]):
                questions.append(q_a['question'])
                answers_start.append(q_a['answers'][0]['answer_start'])
                answers_text.append(q_a['answers'][0]['text'])
            else:
                questions.append([])
                answers_start.append([])
                answers_text.append([])
            contexts.append(sub_para['context'])   
df = pd.DataFrame({1:contexts, 2: questions, 3: answers_text})
for ind,row in df.iterrows():
    sentences = sent_tokenize(row['context'])
    row['question']
    
df.to_csv("/Users/aditya/Downloads/train1.csv", index = None)
dff=df.groupby(1)
df=pd.read_csv("/Users/aditya/Downloads/filteredtrain2.csv",index=None)
paras = list(df["context"].drop_duplicates().reset_index(drop= True))
blob = TextBlob(" ".join(paras))
sentences = [item.raw for item in blob.sentences]
import os

os.chdir('/Users/aditya/anaconda3/envs/squad_bert')
from models import InferSent
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
#a = pickle.load(open('/Users/aditya/encoder/infersent2.pkl','rb'))
infi=InferSent(params_model)
MODEL_PATH = 'encoder/infersent%s.pkl' % 2
#infersent = torch.load('', map_location=lambda storage, loc: storage)
m=infi.load_state_dict(torch.load('/Users/aditya/encoder/infersent1.pkl'))
W2V_PATH = '/Users/aditya/GloVe/glove.840B.300d.txt'
infi.set_w2v_path(W2V_PATH)
infi.build_vocab(sentences, tokenize=True)
dict_embeddings = {}
import time
t=time.time()
for i in range(3619):
    dict_embeddings[sentences[i]] = infi.encode([sentences[i]], tokenize=True)
elap=time.time()-t

questions = list(df["question"])
answers_text = list(df["text"])
answers_start = list(df["answer_start"])
'''pick_out=open("dict_embed.pickle","wb")
pickle.dump(dict_embeddings,pick_out)
pick_out.close()
pick_in=open("dict_embed.pickle","rb")
ex_dict=pickle.load(pick_in)
print(ex_dict['Importing his favorite producers and artists to work on and inspire his recording, West kept engineers behind the boards 24 hours a day and slept only in increments.'][0])
'''
#for i in range(100):
gf=infi.encode(['When did Beyonce start becoming popular?'],tokenize=True)
dict_embeddings[questions[0]] = infi.encode([questions[0]], tokenize=True)
dict_embeddings['Importing his favorite producers and artists to work on and inspire his recording, West kept engineers behind the boards 24 hours a day and slept only in increments.'][0]
df=[]
for i in dict_embeddings:
    df.append(sklearn.metrics.pairwise.cosine_similarity(dict_embeddings[i],gf))
