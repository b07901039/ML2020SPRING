#!/bin/bash
wget https://github.com/b07901039/ML-hw8-seq2seq-model/releases/download/0.0.0/model_final.ckpt
python3 test.py $1 $2
