import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features
import warnings
import random
from torch import randperm
warnings.filterwarnings('ignore')
import torch.nn.functional as F
import scipy.io as scio
import pickle


class Dataset_fMRI(Dataset):
    def __init__(self, root_path,data,shuffle_ix,train_index,test_index,network_id, flag='train'):

        assert flag in ['train', 'test', 'val']  # 断言语句， 如果为真，断言语句不会有任何效果，程序将正常继续执行。
        type_map = {'train': 0, 'val': 1, 'test': 2}  # 将 flag 参数映射到一个特定的整数值  这里没有pred
        self.set_type = type_map[flag]     

        self.root_path = root_path
        self.data = data
      

        self.shuffle_ix = shuffle_ix 
        self.train_index =train_index
        self.test_index= test_index

        self.network_id = network_id

        self.__read_data__()

    def __read_data__(self):

        # load data
        final_data=scio.loadmat(self.root_path)     # TS :  (S, N, T)     

        valid_indices = np.squeeze(final_data['age']) != -1   
        
        try:
            TS = np.array(final_data['timeSeries'])[valid_indices]
        except Exception as e:
            TS = np.array(final_data['ts'])[valid_indices]

        Age = final_data['age'][valid_indices] 


        with open('/data/gyun/project/each_net_roiid.pkl', 'rb') as f:
            each_net_roiid = pickle.load(f)
        try:
            net_roi_index=np.array(each_net_roiid[self.network_id])
            TS = TS[:,net_roi_index,:]
        except Exception as e:
            pass

        # 先打乱数据集，并保持数据和标签的打乱顺序一致，再划分十折
        X_data = TS[self.shuffle_ix]
        Y_data = Age[self.shuffle_ix]
       
        # 每次循环已经划分好了训练集和测试集，shuffle也只是在内部shuffle
        train_X, train_y = X_data[self.train_index], Y_data[self.train_index]
        test_X, test_y  = X_data[self.test_index], Y_data[self.test_index]
        
        # 标准化, 对每个时间点的
        for i in range(train_X.shape[2]):
            scaler = StandardScaler()
            scaler.fit(train_X[:, :, i])
            train_X[:, :, i] = torch.tensor(scaler.transform(train_X[:, :, i]))
            test_X[:, :, i] = torch.tensor(scaler.transform(test_X[:, :, i]))

        pro = int(train_y.shape[0] * 0.9)
        if self.set_type == 0:  # 训练
            X_data = train_X[:pro, :, :]
            Y_data = train_y[:pro]
        elif self.set_type == 1: # 验证
            X_data = train_X[pro:, :, :]
            Y_data = train_y[pro:]
        elif self.set_type == 2: # 测试
            X_data = test_X
            Y_data = test_y


        self.data_x = X_data     # (S, N, T)
        self.data_y = Y_data  # (S,1)

        # self.data_y = np.expand_dims(label_data,axis=-1) # (S,1,1)
 
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_y)

    # def __ymax__(self):
    #     return self.y_max_value,self.y_min_value
