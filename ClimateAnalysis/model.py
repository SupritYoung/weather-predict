from torch import nn
import torch
from d2l import torch as d2l

"""
LSTM 算法模型相关代码
"""

class LSTM(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, num_layers, batch_size, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.num_inputs, self.num_hiddens, self.num_outputs = num_inputs, num_hiddens, num_outputs
        self.num_layers, self.batch_size = num_layers, batch_size
        self.lstm = nn.LSTM(self.num_inputs, self.num_hiddens, self.num_layers, dropout=0.1, batch_first=True)
        # self.init_state = self.init_state(batch_size, num_hiddens, device)
        self.linear = nn.Linear(self.num_hiddens, self.num_outputs) # 这里的 num_outputs 是输出维度，和上面的不同

    def forward(self, X):
        h0 = torch.zeros([self.num_layers, X.shape[0], self.num_hiddens], dtype=torch.double)
        c0 = torch.zeros([self.num_layers, X.shape[0], self.num_hiddens], dtype=torch.double)
        # X = X.view(X.shape[0], -1, self.num_inputs) # 把原有2维度[batch_size, num_inpus]改为3维[batch_size, 1, num_inputs]
        Y, _ = self.lstm.forward(X, (h0.detach(), c0.detach()))
        Y = self.linear(Y[:, -1, :])
        return Y

    # 在初始化函数中,⻓短期记忆网络的隐状态需要返回一个额外的记忆元,单元的值为0,形状为(批量大小, 隐藏单元数)
    # def init_state(self, batch_size, num_hiddens, device):
    #     return (torch.zeros((1, 1, num_hiddens), device=device),
    #             torch.zeros((1, 1, num_hiddens), device=device))