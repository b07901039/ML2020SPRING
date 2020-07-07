# ML HW10 Anomaly Detection 
Distingish unknown images from testing data. 
## Script Usage
```
bash hw10_test.sh <test.npy> <model> <prediction.csv>
```
```
bash hw10_train.sh <train.npy> <model>
``` 
'model' is the path of model,  
if it contains "best", model type will be cnn auto-encoder, pretrained model at "./models/best.pth";  
else if it contains "baseline", model type will be fcn auto-encoder, pretrained model at "./models/baseline.pth".
