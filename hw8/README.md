# ML HW8 Seq2Seq
Translate an english sentence into a chinese sentence.  
* Attention mechanism: use the first layer of decoder hidden vector and use cosine similarity as score funcion.  
* Schedule sampling: use inverse sigmoid funcion.  
* Beam search: beam size = 5.  
## Script Usage  
```
bash hw8_test.sh <data directory> <output path>
```
* Download pretrained model from https://github.com/b07901039/ML-hw8-seq2seq-model/releases/download/0.0.0/model_final.ckpt
* Run test.py  
```
bash hw8_train.sh <data directory>
```
* Run train.py and save model as "model_final.ckpt"  
