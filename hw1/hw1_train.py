"""
select feature by beta_coef, and refill all features < 0 with mean
"""

import sys
import pandas as pd
import numpy as np
import csv
import math
import matplotlib.pyplot as plt

dim = 17 * 9 + 1
# from high to low, len: 9 * 18 = 162
feat_importance = [89, 88, 80, 79, 87, 78, 86, 77, 85, 76, 84, 75, 83, 53, 52, 74, 82, 51, 81, 73, 61, 50, 62, 60, 115, 116, 72, 59, 49, 124, 114, 71, 70, 125, 123, 58, 69, 122, 113, 48, 34, 105, 121, 104, 33, 57, 106, 35, 68, 25, 32, 103, 120, 112, 26, 47, 24, 107, 31, 56, 119, 102, 67, 23, 30, 111, 118, 46, 55, 22, 101, 117, 17, 16, 29, 66, 110, 15, 21, 45, 54, 14, 13, 100, 28, 20, 133, 12, 109, 11, 10, 9, 134, 132, 65, 131, 19, 27, 142, 141, 143, 130, 99, 140, 108, 64, 139, 18, 129, 40, 38, 39, 41, 144, 138, 37, 42, 128, 36, 63, 145, 137, 43, 127, 146, 153, 136, 147, 154, 0, 155, 148, 92, 91, 126, 156, 90, 93, 149, 44, 96, 97, 94, 150, 95, 135, 98, 157, 151, 152, 1, 158, 159, 160, 2, 161, 3, 6, 7, 4, 5, 8]
features = ["AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2",
            "NOx", "O3", "PM10", "PM2.5", "RAINFALL", "RH",
            "SO2", "THC", "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR"]

mean_f = "./drive/My Drive/ML/hw1/mean.npy"
std_f = "./drive/My Drive/ML/hw1/std.npy"

train_f = "./drive/My Drive/ML/hw1/data/train.csv"
test_f = "./drive/My Drive/ML/hw1/data/test.csv"

weight_f = "./drive/My Drive/ML/hw1/weight/weight.npy"
output_f = "./drive/My Drive/ML/hw1/predict/output_select50_pow4_fillAll0.csv"

def read_data():
    
    data = pd.read_csv(train_f, encoding = 'big5')
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy() # 4320 * 24

    return raw_data

def preprocess(raw_data, index = 50, power = 4):

    # 5760 * 18, row: timestamp, col: feature
    raw_data = raw_data.reshape(-1, 18, 24).transpose(0, 2, 1).reshape(-1, 18).astype("float")
    raw_data = pd.DataFrame(data = raw_data, columns = features)
    raw_data[raw_data < 0] = 0
    

    # 4320 * 24, convert back to the structure of raw_data
    raw_data = raw_data.to_numpy()
    raw_data = raw_data.transpose(1, 0).reshape(18, -1, 24).transpose(1, 0, 2).reshape(-1, 24)
    month_data = {}
    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
        month_data[month] = sample

    x = np.empty([12 * 471, 18 * 9], dtype = float)
    y = np.empty([12 * 471, 1], dtype = float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
    

    # feature engineering
    delete_list = feat_importance[index:]
    power_list = feat_importance[:power]
    for i in range(len(power_list)):
        x = np.concatenate((x, np.power(x[:,power_list[i]].reshape(-1, 1).astype(float), 2)), axis = 1).astype(float)
    x = np.delete(x, delete_list, axis = 1)
    
    # normalize
    mean_x = np.mean(x, axis = 0) #18 * 9 
    std_x = np.std(x, axis = 0) #18 * 9 
    for i in range(len(x)): #12 * 471
        for j in range(len(x[0])): #18 * 9 
            if std_x[j] != 0:
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    np.save(mean_f, mean_x)
    np.save(std_f, std_x)
    
    return x, y, mean_x, std_x

def train(x, y):
    # training
    loss_list = list()
    dim = x.shape[1] + 1
    w = np.zeros([dim, 1])
    x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
    learning_rate = 10 #
    iter_time = 5500 #
    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
        if(t%10==0 and t > 10):
            print(str(t) + ":" + str(loss))
            loss_list.append(loss)
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

    np.save(weight_f, w)
    # loss_np = np.asarray(loss_list, dtype=np.float32)
    # np.savetxt("feat_final_train.txt", loss_np)


def train_validation(x, y):
    
    loss_list = list()
    dim = x.shape[1] + 1
    w = np.zeros([dim, 1])
    x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
    
    # split train / validation set
    x_train = x[: math.floor(len(x) * 0.8), :]
    y_train = y[: math.floor(len(y) * 0.8), :]
    
    x_validation = x[math.floor(len(x) * 0.8): , :]
    y_validation = y[math.floor(len(y) * 0.8): , :]

    learning_rate = 10 #
    iter_time = 3000 #
    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001

    validation_loss = list()
    train_loss = list()
    _iter = list()

    w = np.zeros([dim, 1])
    for t in range(iter_time):
        loss_t = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train, 2))/471/12/0.8)#rmse
        loss_v = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/471/12/0.2)#rmse
        if(t%10==0) and t > 10:
            validation_loss.append(loss_v)
            train_loss.append(loss_t)
            _iter.append(t)
            print("{}: training loss: {}, validation loss: {}".format(t, train_loss[-1], validation_loss[-1]))


        gradient = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train) #dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    print("Total training loss: {}, validation loss: {}".format(train_loss[-1], validation_loss[-1]))
    # return train_loss[-1], validation_loss[-1]
    # validation_lost = np.asarray(validation_lost, dtype = np.float32)
    # np.savetxt("feature_final.txt", validation_lost)
    
    # plt.plot(_iter, train_loss)
    # plt.plot(_iter, validation_loss)
    # plt.legend(["train", "validation"])
    # plt.title("loss")
    # plt.xlabel("iter")
    # plt.ylabel("loss")
    # plt.savefig("./drive/My Drive/ML/hw1/pic/final.png")
    # plt.show()
    
    np.save(weight_f, w)




def test(mean_x, std_x, index = 50, power = 4):
    # read test data
    testdata = pd.read_csv(test_f, header = None, encoding = 'big5')
    test_data = testdata.iloc[:, 2:]
    test_data[test_data == 'NR'] = 0
    test_data = test_data.to_numpy()

    # row: timestamp, col: feature
    test_data = test_data.reshape(-1, 18, 9).transpose(0, 2, 1).reshape(-1, 18).astype("float")
    test_data = pd.DataFrame(data = test_data, columns = features)
    test_data[test_data < 0] = 0
    
    
    # 240 * 9, convert back to the structure of raw_data
    test_data = test_data.to_numpy()
    test_data = test_data.transpose(1, 0).reshape(18, -1, 9).transpose(1, 0, 2).reshape(-1, 9)

    test_x = np.empty([240, 18*9], dtype = float)
    for i in range(240):
        test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
    
    # feature engineering
    delete_list = feat_importance[index:]
    power_list = feat_importance[:power]
    for i in range(len(power_list)):
        test_x = np.concatenate((test_x, np.power(test_x[:,power_list[i]].reshape(-1, 1).astype(float), 2)), axis = 1).astype(float)
    test_x = np.delete(test_x, delete_list, axis = 1)
    
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
    
   

    w = np.load(weight_f)
    ans_y = np.dot(test_x, w)

    # write file
    with open(output_f, mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)

if __name__ == "__main__":

    """
    preprocess(), test() includes selecting features
    """
    raw_data = read_data()
    x, y, mean, std = preprocess(raw_data, 50, 4)
    train(x, y)
    # train_validation(x, y)
   
    test(mean, std, 50, 4)
    

