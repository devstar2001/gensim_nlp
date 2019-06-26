#%%
import glob
import os
import random
import logging
import pandas as pd
import numpy
import pickle
import gensim
import operator
import re
import numpy as np
import pickle as pk
from random import shuffle

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from typing import Iterable

from gensim.models.word2vec import Word2Vec
from nltk import tokenize, casual_tokenize
import nltk
from zipfile import ZipFile
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from nltk.corpus import stopwords
#%%
from gensim.models import KeyedVectors
#%%
# os.chdir('/Users/philine/Dropbox/sicss/sif_code/')
#%%
from gensim.scripts.glove2word2vec import glove2word2vec


model_path = './models/times_of_india.model'

#%%
from gensim.models import Word2Vec

model = Word2Vec.load(model_path)



#%%
word_count_dict = {}
#%%
for word, vocab_obj in model.wv.vocab.items():
    word_count_dict[word] = vocab_obj.count
#%%
"""We first create a dictionary that maps from a word to its frequency and then use the frequency of
a word to compute its sif-weight (saved in sif_dict)."""

sif_dict = {}
#%%
for word, count in word_count_dict.items():
    sif_dict[word] = .001/(.001+count)
#%%
pk.dump(sif_dict, open("sif_dict.p", "wb"))
#%%
# Write some code to load sample data and split into sentences!


### CODE HERE ###

df = pd.read_csv('./sample_data/output_2012-1-1_clean.csv', quotechar='"', error_bad_lines=False)
items = df.values


#%%
#removing stopword

#defines a tokenizer that does not save punctuation and convert everything to lower case
#also removes stopwords
stop = stopwords.words('english')
stop = [x.lower() for x in stop]
stop = set(stop)
#%%
def tokenize_no_punct_all_lower(txt):
    txt_tokenize = casual_tokenize(txt,preserve_case=False,strip_handles=True)
    txt_tokenize = [word for word in txt_tokenize if re.sub(r"\-", "", word).isalpha()]
    txt_tokenize = [word for word in txt_tokenize if word not in stop]
    return txt_tokenize
#%%
vocab_set = set(model.wv.vocab.keys())
#%%
def document_lines_vec(txt):
    # Text should be the sentences (str type we loaded from the sample data above)

    tokens = tokenize_no_punct_all_lower(txt)
    tokens_tagged = nltk.pos_tag(tokens,tagset='universal')
    tokens = [i[0] for i in tokens_tagged if i[1] in ["VERB","NOUN","ADJ","ADV"]]
    tokens = [t for t in tokens if t in vocab_set]
    sif_vec = np.mean([sif_dict[t]*model.wv[t] for t in tokens],axis=0)
    return sif_vec
#%%
sif_vec_dict = {}
#%%
for item in items:
    sentences = tokenize.sent_tokenize(item[2])
    sif_vec_dict[item[2].split('/')[-1]] = [document_lines_vec(sent) for sent in sentences]
#%%
pk.dump(sif_vec_dict, open("sif_vec_dict.p", "wb"))
print("saved")
#%%
