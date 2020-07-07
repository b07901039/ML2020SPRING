#!/bin/bash
wget https://github.com/b07901039/ML-hw4-model/releases/download/0.0.0/w2v_all.model
wget https://github.com/b07901039/ML-hw4-model/releases/download/0.0.0/w2v_all.model.trainables.syn1neg.npy
wget https://github.com/b07901039/ML-hw4-model/releases/download/0.0.0/w2v_all.model.wv.vectors.npy
wget https://github.com/b07901039/ML-hw4-model/releases/download/0.0.1/ckpt_final.model
python3 test.py $1 $2
