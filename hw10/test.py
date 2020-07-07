import sys
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from scipy.cluster.vq import vq, kmeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import numpy as np
from model import fcn_autoencoder
from cnn2 import conv2_autoencoder


def knn(y, n): # y is encoder output, n is num of clusters
  y = y.reshape(len(y), -1)
  scores = list()
  kmeans_y = MiniBatchKMeans(n_clusters=n, batch_size=100).fit(y)
  y_cluster = kmeans_y.predict(y)
  y_dist = np.sum(np.square(kmeans_y.cluster_centers_[y_cluster] - y), axis=1)
  y_pred = y_dist
  return y_pred

def pca(y, n): # y is encoder output, n is num of clusters
  y = y.reshape(len(y), -1)
  pca = PCA(n_components=n).fit(y)
  y_projected = pca.transform(y)
  y_reconstructed = pca.inverse_transform(y_projected)  
  dist = np.sqrt(np.sum(np.square(y_reconstructed - y).reshape(len(y), -1), axis=1))
  y_pred = dist
  return y_pred

if __name__ == "__main__":
    
    batch_size = 128

    data_pth = sys.argv[1]
    model_pth = sys.argv[2]
    output_pth = sys.argv[3]

    if "baseline" in model_pth:
      model_type = "fcn"
    elif "best" in model_pth:
      model_type = "cnn"
    else:
      print("unknown model type")

    test = np.load(data_pth, allow_pickle=True)
    
    if model_type == 'fcn':
        y = test.reshape(len(test), -1)
    else:
        y = test
        
    data = torch.tensor(y, dtype=torch.float)
    test_dataset = TensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    
    model = torch.load(model_pth, map_location='cuda')

    model.eval()
    reconstructed = list()
    for i, data in enumerate(test_dataloader): 
        
        if model_type == 'cnn':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        
        output = model(img)
        
        if model_type == 'cnn':
            output = output.transpose(3, 1)

        reconstructed.append(output.cpu().detach().numpy())

    reconstructed = np.concatenate(reconstructed, axis=0)

    
    anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
    y_pred = anomality
    
    with open(output_pth, 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))
