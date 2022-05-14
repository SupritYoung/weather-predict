import torch
import data
from d2l import torch as d2l
from torch import nn
import model
import numpy as np
from matplotlib import pyplot as plt
import config

# 梯度剪裁
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train(model, iterator, optimizer, loss, device):
    model.train()
    train_loss = 0

    for X, y in train_iter:
        # move to GPU if necessary
        X, y = X.to(device), y.to(device)

        # generate prediction
        optimizer.zero_grad()
        preds = model(X)
        preds = preds.view(-1)

        # calculate loss
        l = loss(preds, y)

        # compute gradients and update weights
        l.backward()
        optimizer.step()

        # record training losses
        train_loss += l.item()

    # print completed result
    print('train_loss: %f' % (train_loss))
    return train_loss


def test(model, test_iter, loss, device):
    model.eval()  # Enter Evaluation Mode
    test_loss = 0

    with torch.no_grad():
        for X, y in test_iter:
            # move to GPU if necessary
            X, y = X.to(device), y.to(device)

            # generate prediction
            preds = model(X)
            preds = preds.view(-1)

            # convert target tensor to long
            y = y.long()

            # calculate loss
            l = loss(preds, y)

            # record training losses
            test_loss += l.item()

    # print completed result
    print('test_loss: %s' % (test_loss))
    return test_loss

if __name__ == '__main__':
    # 读取数据
    batch_size = 32
    train_iter, _ = data.load_data(batch_size, train_rate=0.6, train_data=True)

    # 定义模型参数
    num_hiddens, num_layers = 1024, 3           # 超参数
    # num_inputs = 200  # 输入维度为温度的范围，我们假定温度只有整数且为 (-100, 100)
    num_inputs ,num_outputs = 1, 1       # 因为 LSTM 的输入为序列模型，而我们输入的仅有天气数据，所以输入维度为 1
    epochs, lr = config.epochs, config.lr     # 训练轮数、学习率
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 定义模型
    net = model.LSTM(num_inputs, num_hiddens, num_outputs, num_layers, batch_size)
    net = net.double()
    print(net)
    # 调用梯度下降方法
    optimizer = torch.optim.Adam(net.parameters(), lr)
    # 定义损失函数
    loss = nn.MSELoss()

    train_loss = []
    for epoch in range(epochs):
        print("===== Epoch %i =====" % epoch)
        train_loss.append(train(net, train_iter, optimizer, loss, device))

    torch.save(net, 'saved_model/lstm.pt')

    # 可视化 loss
    d2l.plot(np.arange(len(train_loss), dtype='int32'), train_loss, 'epoch', 'train loss', figsize=(6, 5))
    plt.title('Train Loss', fontsize=18)
    plt.show()


    # model.train()
    # for epoch in range(num_epochs):
    #     timer = d2l.Timer()
    #     metric = d2l.Accumulator(1)
    #     for X, y in train_iter:
    #         y_hat = net(X)
    #         optimizer.zero_grad()
    #         l = loss(y_hat.float(), y.float())
    #         l.sum().backward()
    #         grad_clipping(net, 1)
    #         optimizer.step()
    #         # with torch.no_grad():
    #         #     metric.add(l.sum())
    #     print(f'epoch {epoch + 1}, loss {l.sum():f}')
    #     # if (epoch + 1) % 10 == 0:
    #     #     animator.add(epoch + 1, metric[0])
