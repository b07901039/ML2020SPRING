hw3_test.sh:  
download model from https://github.com/b07901039/ML-hw3-model/releases/download/0.0.0/model.pkl, and save the model at [./model.pkl]  
run test.py  

test.py:  
read testing set from [data directory]  
load model form [./model.pkl] (download by hw3_test.sh)  
make predictions and output at [prediction file]  

hw3_train.sh:  
run train.py  

train.py:  
read training and validation set from [data directory]  
train model  
save model at [./model.pkl]  
