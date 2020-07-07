import sys
import numpy as np
import pandas as pd
import csv

test_f = sys.argv[1]
output_f = sys.argv[2]

mean_f = "./mean.npy"
std_f = "./std.npy"
weight_f = "./weight_best_final.npy"

index = 50
power = 4
feat_importance = [89, 88, 80, 79, 87, 78, 86, 77, 85, 76, 84, 75, 83, 53, 52, 74, 82, 51, 81, 73, 61, 50, 62, 60, 115, 116, 72, 59, 49, 124, 114, 71, 70, 125, 123, 58, 69, 122, 113, 48, 34, 105, 121, 104, 33, 57, 106, 35, 68, 25, 32, 103, 120, 112, 26, 47, 24, 107, 31, 56, 119, 102, 67, 23, 30, 111, 118, 46, 55, 22, 101, 117, 17, 16, 29, 66, 110, 15, 21, 45, 54, 14, 13, 100, 28, 20, 133, 12, 109, 11, 10, 9, 134, 132, 65, 131, 19, 27, 142, 141, 143, 130, 99, 140, 108, 64, 139, 18, 129, 40, 38, 39, 41, 144, 138, 37, 42, 128, 36, 63, 145, 137, 43, 127, 146, 153, 136, 147, 154, 0, 155, 148, 92, 91, 126, 156, 90, 93, 149, 44, 96, 97, 94, 150, 95, 135, 98, 157, 151, 152, 1, 158, 159, 160, 2, 161, 3, 6, 7, 4, 5, 8]
features = ["AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2",
            "NOx", "O3", "PM10", "PM2.5", "RAINFALL", "RH",
            "SO2", "THC", "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR"]
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

mean_x = np.load(mean_f)
std_x = np.load(std_f)

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