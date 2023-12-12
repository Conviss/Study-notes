import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class GRU(nn.Module):

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向GRU

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # utilize the GRU model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        h_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.gru(input_seq, (h_0.detach()))
        pred = self.linear1(output[:, -1, :])
        return pred.view(-1)


def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return torch.tensor(np.array(dataX)), torch.tensor(np.array(dataY))


device = torch.device("cpu")

if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# parse_dates 表示将某一列设置为 时间类型
data = pd.read_csv("data/Stock data/train.csv",parse_dates=["Date"],index_col=[0])
print(data.shape)

data = np.array(data).astype('float32')
print(data)

data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
print(data)

test_split = round(len(data) * 0.20)
data_train = data[:-test_split]
data_test = data[-test_split:]
print(data_train.shape)
print(data_test.shape)

train_x_tensor, train_y_tensor = createXY(data_train, 30)

print("train_x_tensor Shape-- ", train_x_tensor.shape)
print("train_y_tensor Shape-- ", train_y_tensor.shape)

gru = GRU(train_x_tensor.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(gru.parameters(), lr=1e-2)

prev_loss = 1000
max_epochs = 2000

train_x_tensor = train_x_tensor.to(device)

for epoch in range(max_epochs):
    output = gru(train_x_tensor).to(device)
    loss = criterion(output, train_y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if loss < prev_loss:
    #     torch.save(gru.state_dict(), 'gru_model.pt')  # save model parameters to files
    #     prev_loss = loss

    if loss.item() < 1e-4:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
        print("The loss value is reached")
        break

    elif (epoch + 1) % 100 == 0:
        print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

gru_model = gru.eval()  # switch to testing model

# prediction on test dataset
test_x_tensor, test_y_tensor = createXY(data_test, 30)

# print(test_x_tensor)
test_x_tensor = test_x_tensor.to(device)

pred_y_for_test = gru(test_x_tensor).to(device)

loss = criterion(pred_y_for_test, test_y_tensor)
print("test loss：", loss.item())

# ----------------- plot -------------------
plt.figure()

plt.plot(test_y_tensor.detach().numpy(), label='y_tst')
plt.plot(pred_y_for_test.detach().numpy(), label='pre_tst')

plt.xlabel('t')
plt.ylabel('Stock_data')
plt.show()