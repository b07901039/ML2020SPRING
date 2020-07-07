# ML hw7 Network Compressing  
use small model (<=300k bytes) to accomplish hw3 task  
(cnn image classification)
## Script Usage
```
bash  hw7_test.sh  <data directory>  <prediction file>  
```
* use pretrained small model (model.pkl) to run test.py 
* save output at <prediction file>
```
bash hw7_train.sh <data directory>  
```
* download pretrained big model (teacher net) from https://github.com/b07901039/ML-hw7-teachernet/releases/download/0.0.0/teacher_resnet18.bin  
* run train.py, use teacher net to train a big model (1 MB) and save as big_model.bin  
* run weightQ.py, compress big_model.bin into model.pkl (285 KB)  
