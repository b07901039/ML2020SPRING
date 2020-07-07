"""
train word embedding
"""
import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec

from utils import load_training_data, load_testing_data

def train_word2vec(x, size, window):
    # 訓練 word to vector 的 word embedding
    # size: dim of word, iter: iteration, sg: 0 (CBOW), 1 (Skip-gram), windows: # of words taken to predict the word, min_count:  Ignores all words with total frequency lower than this 
    model = word2vec.Word2Vec(x, size=size, window=window, min_count=5, workers=12, iter=10, sg=1) # size=250, window=5, min_count=5, workers=12, iter=10, sg=1
    return model

if __name__ == "__main__":
    
    print("loading training data ...")
    train_x, y = load_training_data('./data/training_label.txt')
    train_x_no_label = load_training_data('./data/training_nolabel.txt')

    print("loading testing data ...")
    test_x = load_testing_data('./data/testing_data.txt')

    model = train_word2vec((train_x + train_x_no_label + test_x), 5, 250)
    # model = train_word2vec(train_x + test_x)
    
    print("saving model ...")
    model_f = './w2v_all.model'
    model.save(model_f)

