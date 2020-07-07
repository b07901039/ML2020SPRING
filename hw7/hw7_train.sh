#!/bin/bash
wget https://github.com/b07901039/ML-hw7-teachernet/releases/download/0.0.0/teacher_resnet18.bin
python3 train.py $1
python3 weightQ.py
