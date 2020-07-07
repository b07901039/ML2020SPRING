"""
logistic regression, add square, tri for all continuous features, add regularization
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import pandas as pd



np.random.seed(0)
X_train_fpath = './drive/My Drive/ML/hw2/data/X_train'
Y_train_fpath = './drive/My Drive/ML/hw2/data/Y_train'
X_test_fpath = './drive/My Drive/ML/hw2/data/X_test'

msg = "sqTriAll_drop_1e-02_regu0_nodum"
output_fpath = './drive/My Drive/ML/hw2/predict/output_{}.csv'.format(msg)
loss_f = "./drive/My Drive/ML/hw2/pic/loss_{}.png".format(msg)
acc_f = "./drive/My Drive/ML/hw2/pic/acc_{}.png".format(msg)

# weight_f = "./drive/My Drive/ML/hw2/weight_final.npy"
# b_f = "./drive/My Drive/ML/hw2/b_final.npy"




def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

def plot(train_loss, dev_loss, train_acc, dev_acc, loss_f = "loss.png", acc_f = "acc.png"):

    # Loss curve
    plt.plot(train_loss)
    plt.plot(dev_loss)
    plt.title('Loss')
    plt.legend(['train', 'dev'])
    plt.savefig(loss_f)
    plt.show()

    # Accuracy curve
    plt.plot(train_acc)
    plt.plot(dev_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'dev'])
    plt.savefig(acc_f)
    plt.show()

def read_file():
  # Parse csv files to numpy array
    with open(X_train_fpath) as f:
        next(f)
        X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    with open(Y_train_fpath) as f:
        next(f)
        Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
    with open(X_test_fpath) as f:
        next(f)
        X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    return X_train, Y_train, X_test

def preprocess(X_train, X_test):
  
  # feature selection
  with open(X_test_fpath) as f:
        content = f.readline().strip('\n').split(',')
  features = np.array(content[1:])
  X_train = pd.DataFrame(data = X_train, columns = features)
  X_test = pd.DataFrame(data = X_test, columns = features)

  # add square and tri terms
  continuous_feat = ['age',
      'wage per hour',
      'capital gains',
      'capital losses',
      'dividends from stocks',
      'num persons worked for employer',
      'weeks worked in year']
  
  for feat in continuous_feat:
    X_train["{}**2".format(feat)] = X_train[feat] ** 2
    X_test["{}**2".format(feat)] = X_test[feat] ** 2
    X_train["{}**3".format(feat)] = X_train[feat] ** 3
    X_test["{}**3".format(feat)] = X_test[feat] ** 3
  X_train["wage per hour * weeks worked in year"] = X_train["wage per hour"] * X_train["weeks worked in year"]
  X_test["wage per hour * weeks worked in year"] = X_test["wage per hour"] * X_test["weeks worked in year"]
  
  X_train = X_train.to_numpy()
  X_test = X_test.to_numpy()
  
  # Normalize training and testing data
  X_train, X_mean, X_std = _normalize(X_train, train = True)
  X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
  np.save("./drive/My Drive/ML/hw2/mean.npy", X_mean)
  np.save("./drive/My Drive/ML/hw2/std.npy", X_std)
  return X_train, X_test

def main(X_train, Y_train, X_test):

    # Split data into training set and development set
    dev_ratio = 0.2

    X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)
    

    train_size = X_train.shape[0]
    dev_size = X_dev.shape[0]
    test_size = X_test.shape[0]
    data_dim = X_train.shape[1]
    print('Size of training set: {}'.format(train_size)) # 48830
    print('Size of development set: {}'.format(dev_size)) # 5426
    print('Size of testing set: {}'.format(test_size)) # 27622
    print('Dimension of data: {}'.format(data_dim)) # 510

    # training
    # Zero initialization for weights ans bias
    w = np.zeros((data_dim,)) 
    b = np.zeros((1,))

    # Some parameters for training    
    max_iter = 500
    batch_size = 8
    learning_rate = 0.015
    _lambda = 10**-4 # regularization

    # Keep the loss and accuracy at every iteration for plotting
    train_loss = []
    dev_loss = []
    train_acc = []
    dev_acc = []

    # Calcuate the number of parameter updates
    step = 1

    # Iterative training
    for epoch in range(max_iter):
        # Random shuffle at the begging of each epoch
        X_train, Y_train = _shuffle(X_train, Y_train)
            
        # Mini-batch training
        for idx in range(int(np.floor(train_size / batch_size))):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            # Compute the gradient
            w_grad, b_grad = _gradient(X, Y, w, b)
                
            # gradient descent update
            # learning rate decay with time
            w = w - learning_rate/np.sqrt(step) * (w_grad + _lambda * w)
            b = b - learning_rate/np.sqrt(step) * b_grad

            step = step + 1
                
        Compute loss and accuracy of training set and development set
        y_train_pred = _f(X_train, w, b)
        Y_train_pred = np.round(y_train_pred)
        train_acc.append(_accuracy(Y_train_pred, Y_train))
        train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

        y_dev_pred = _f(X_dev, w, b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
        dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

        print("%d: train acc = %.4f, dev acc = %.4f" % (epoch, train_acc[-1], dev_acc[-1]))
    print('Training loss: {}'.format(train_loss[-1]))
    print('Development loss: {}'.format(dev_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Development accuracy: {}'.format(dev_acc[-1]))
    
    # plot(train_loss, dev_loss, train_acc, dev_acc, loss_f, acc_f)
    return w, b

def test(X_test, w, b):
    # Predict testing labels
    predictions = _predict(X_test, w, b)
    with open(output_fpath, 'w') as f:
        f.write('id,label\n')
        for i, label in  enumerate(predictions):
            f.write('{},{}\n'.format(i, label))
  
if __name__ == "__main__":
  X_train, Y_train, X_test = read_file()
  X_train, X_test = preprocess(X_train, X_test)
  w_list = []
  b_list = []
  for i in range(5):
    print("time: {}".format(i))
    w, b = main(X_train, Y_train, X_test)
    w_list.append(w)
    b_list.append(b)
  w = np.sum(w_list, axis = 0) / 5
  b = np.sum(b_list) / 5
  # np.save(weight_f, w)
  # np.save(b_f, b)
  test(X_test, w, b)

  
