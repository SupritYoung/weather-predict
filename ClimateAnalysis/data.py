import torch
import pandas as pd
import re
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import dataset as ds
import config

""""
数据相关的一些方法
"""

# 加载数据，划分训练集、测试集
# batch_size: 批量大小
# train_rate：训练集比例
def load_data(batch_size=32, train_rate = 0.6, train_data = True):
    process_data()
    dataset = ds.TemperatureDataset(config.data, '平均气温')

    # 随机划分训练集和测试集
    train_len = int(train_rate * len(dataset))
    valid_len = len(dataset) - train_len
    TrainData, ValidationData = random_split(dataset, [train_len, valid_len])

    print(config.city_name + " 天气数据的数据总数：", len(dataset))
    # Load into Iterator (each time get one batch)
    if train_data is True:
        loader = DataLoader(TrainData, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        print(config.city_name + " 天气数据的训练集大小：", len(TrainData))
    else:
        loader = DataLoader(ValidationData, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        print(config.city_name + " 天气数据的验证集大小：", len(ValidationData))

    return loader, dataset

# 生成自回归的特征和标签
# tau：嵌入维度，要考虑初始化问题放弃后 tau 个数据
# 特征：第 i 个数据的前 tau 个数据构成的序列
# 标签：第 i 个特征下一个数据
def get_features_labels(x, tau = 4):
    tau = 4
    T = len(x)+1   # 总点数
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T-tau+i]
    labels = x[tau:].reshape((-1, 1))
    return features, labels

# 处理气温的格式，添加最低气温、最高气温、平均气温
def process_data():
    # 读取一开始爬取的原始表
    weather_data = pd.read_excel('data/weather_data_ori.xlsx', sheet_name=None)
    writer = pd.ExcelWriter('data/weather_data.xlsx')

    # 遍历 excel 中的所有 sheet
    for sheet_name, sheet in weather_data.items():
        # 处理过后的新表（为方便后面训练，主要体现出气温相关的特征）
        output_data = pd.DataFrame(columns=['日期', '天气', '最低气温', '最高气温', '平均气温'])
        output_data['日期'] = sheet['日期']
        output_data['天气'] = sheet['天气状况']
        output_data['最低气温'] = sheet['气温'].map(lambda s: re.findall(re.compile(u'^.*?(?=℃)'), s)[0])
        output_data['最高气温'] = sheet['气温'].map(lambda s: re.findall(re.compile(u'(?<=/).*?(?=℃)'), s)[0])

        output_data['平均气温'] = output_data.apply(lambda x: (int(x['最低气温'])+int(x['最高气温'])) // 2, axis=1)
        output_data.to_excel(writer, index_label='index', sheet_name=sheet_name)
    writer.save()
    # return output_data


if __name__ == '__main__':
    load_data()
