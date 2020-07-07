import sys
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

sys.path.append("./src")
from utils import load_testing_data
from preprocess import Preprocess
from model import bi_LSTM_Net
from data import TwitterDataset
sys.path.pop()

def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1 # 大於等於 0.5 為負面
            outputs[outputs<0.5] = 0 # 小於 0.5 為正面
            ret_output += outputs.int().tolist()
    
    return ret_output

if __name__ == "__main__":
    
    w2v_path = './w2v_all.model'
    model_f = "./ckpt_final.model"

    testing_data = sys.argv[1]
    output_f = sys.argv[2]

    sen_len = 35 # 20
    fix_embedding = True # fix embedding during training
    batch_size = 128

    # 開始測試模型並做預測
    # print("loading testing data ...")
    test_x = load_testing_data(testing_data)
    
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 8)
    # print('\nload model ...')
    model = torch.load(model_f)
    outputs = testing(batch_size, test_loader, model, 'cuda')

    # 寫到 csv 檔案供上傳 Kaggle
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
    # print("save csv ...")
    tmp.to_csv(output_f, index=False)
    # print("Finish Predicting")

