#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:36:10 2019

@author: aditya
"""

dff=df.groupby('1')
cos_mat2={}
for i in range(1,28):
    cos_mat2[i]=[] 
cos_mat2['target']=[] 
dict_embeddings1={}
counter=0
import sklearn.metrics as sm
for context, group in dff:
    if counter<4:        
        qus = group['2']
        ans = group['3'].reset_index(drop=True)
        para = context
        i=0
        blob=TextBlob("".join(para))
        sent=[item.raw for item in blob.sentences]
        qus_vector=infi.encode(qus,tokenize=True)
        dict_embeddings1=infi.encode(sent,tokenize=True)
        for j in range(len(qus_vector)):
            for i in range(27):
                if i>=dict_embeddings1.shape[0]:
                    cos_mat2[i+1].append(float(0))
                else:
                    cos_mat2[i+1].append(metrics.pairwise.cosine_similarity([dict_embeddings1[i]],[qus_vector[j]]))
                    if ans[j] in sent[i]:
                        target = i+1
            cos_mat2['target'].append(target)
        counter+=1
    else:
        break
pick_out=open("cos_matrix2.pickle","wb")
pickle.dump(cos_mat1,pick_out)
pick_out.close()
pick_out=open("cos_matrixfin.pickle","wb")
pickle.dump(cos_mat1,pick_out)
pick_out.close()        
#pick_in=open("cos_matrix.pickle","rb")
#ex_dict=pickle.load(pick_in)    