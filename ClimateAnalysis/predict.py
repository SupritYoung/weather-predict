import torch
import data
import numpy as np
import config
from matplotlib import pyplot as plt

def predict(model, device, data):
    model.eval()  # Enter Evaluation Mode
    with torch.no_grad():
        pred = model(data)
        return pred

if __name__ == '__main__':
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = torch.load('saved_model/lstm.pt')

    test_iter, dataset = data.load_data(batch_size, train_rate=0.6, train_data=False)
    # 连续预测接下来 N 天的温度
    N = 5
    # 随机抽取 P 个点作为测试集
    P = 50
    points = np.random.randint(dataset.sample_len, high=len(dataset), size=P)
    preds = np.zeros((P, N))
    # 预测从这 P 个点开始接下来 N 天的温度
    for idx_i, i in enumerate(points):
        normalized_temp, target = dataset[i]
        temperatures = normalized_temp

        for idx_j, j in enumerate(range(N)):
            ii = temperatures.view(1, dataset.sample_len + idx_j, 1)  # transform into (batch, sequence, features)
            pred = predict(net, device, ii)
            preds[idx_i][idx_j] = pred

            # remove first temp and append predicted data at last
            # temperatures = torch.cat((temperatures[1:], pred.view(-1)))
            temperatures = torch.cat((temperatures, pred))

        # Inverse transform the predictions
        actual_predictions = config.scaler.inverse_transform(preds[idx_i].reshape(-1, 1))
        preds[idx_i] = actual_predictions.reshape(-1)
        print("Predicts: ", actual_predictions)
        print("Expected: ", config.scaler.inverse_transform(target.reshape(-1, 1)))

    # 预测结果可视化
    # Data
    months = range(0, len(config.data['日期']))

    # Draw Original Lines
    plt.figure(figsize=(30, 10))
    plt.title('Temperature Prediction in '+config.city_name, fontsize=18)
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.plot(months, config.data['平均气温'], color='b')

    # Draw all predicted points
    for idx_i, i in enumerate(points):
        i_months = range(i, i + N)
        i_temps = preds[idx_i]
        plt.plot(i_months, i_temps, color='r', linewidth=5)
        print("Predicted months: ", i_months)

    plt.show()

    # y_preds = np.ndarray(batch_size)
    # y_actuals = np.ndarray(batch_size)
    # 在测试集上进行预测
    # for X, y in test_iter:
    #     y_pred = predict(net, device, X)
    #     np.append(y_preds, y_pred)
    #     np.append(y_actuals, y)
    #     y_preds = dataset.scaler.inverse_transform(y_preds.reshape(-1, 1))
    #     y_actuals = dataset.scaler.inverse_transform(y_actuals.reshape(-1, 1))
    #     print("Predicts: ", int(y_preds))
    #     print("Expected: ", int(y_actuals))