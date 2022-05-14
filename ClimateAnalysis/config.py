from sklearn.preprocessing import MinMaxScaler
import pandas as pd

"""
全局配置相关的一些信息
"""

# ML Parameters
lr = 0.003
epochs = 500
batch_size = 32

# Normalization
scaler = MinMaxScaler(feature_range=(-1, 1))

# 全局定义要训练和预测的城市名称
city_name = 'beijing' # shanghai guangzhou zhengzhou
data = pd.read_excel('data/weather_data.xlsx', sheet_name=city_name)
