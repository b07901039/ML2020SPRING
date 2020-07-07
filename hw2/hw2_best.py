import sys
import numpy as np
import pandas as pd

X_test_fpath = sys.argv[5]
output_fpath = sys.argv[6]

weight_f = "./weight_final.npy"
b_f = "./b_final.npy"
mean_f = "./mean.npy"
std_f = "./std.npy"

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

if __name__ == "__main__":
    with open(X_test_fpath) as f:
        next(f)
        X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    
    # feature selection
    with open(X_test_fpath) as f:
        content = f.readline().strip('\n').split(',')
    features = np.array(content[1:])
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
        X_test["{}**2".format(feat)] = X_test[feat] ** 2
        X_test["{}**3".format(feat)] = X_test[feat] ** 3
    X_test["wage per hour * weeks worked in year"] = X_test["wage per hour"] * X_test["weeks worked in year"]

    X_test = X_test.to_numpy()

    Ｘ_mean = np.load(mean_f)
    X_std = np.load(std_f)
    
    # Normalize training and testing data
    X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

    # Predict testing labels
    w = np.load(weight_f)
    b = np.load(b_f)
    
    predictions = _predict(X_test, w, b)
    with open(output_fpath, 'w') as f:
        f.write('id,label\n')
        for i, label in  enumerate(predictions):
            f.write('{},{}\n'.format(i, label))


  
