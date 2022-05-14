import pandas as pd
import numpy as np
import torch.utils.data as data
import torch
import config

class TemperatureDataset(data.Dataset):

    def __init__(self, data, name):
        # 读取原始数据集
        self.orig_dataset = data[name].to_numpy()
        
        # 正则化数据集
        self.normalized_dataset = np.copy(self.orig_dataset)
        self.normalized_dataset = self.normalized_dataset.reshape(-1, 1)
        self.normalized_dataset = config.scaler.fit_transform(self.normalized_dataset)
        self.normalized_dataset = self.normalized_dataset.reshape(-1)

        # 使用 sample_len 个过去的天气预测下一个天气，sample_len == tau
        self.sample_len = 30

    def __len__(self):
        if len(self.orig_dataset) > self.sample_len:
            return len(self.orig_dataset) - self.sample_len
        else:
            return 0

    def __getitem__(self, idx):

        # y is the last records
        y = self.normalized_dataset[self.sample_len + idx]
        y = np.array(y).astype(np.float64)  # convert to numpy array

        # data is the previous five records
        X = self.normalized_dataset[idx:(idx + self.sample_len)]

        # LSTM requires time-series data to shape in this way
        # LSTM 需要时间序列形式的数据
        X = X.reshape((-1, 1))

        # convert to PyTorch tensors
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        return X, y
