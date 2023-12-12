import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
from scipy.stats import genpareto
from torch import distributions
from torch.utils.data import DataLoader

def pot_analysis(data, threshold, q):
    # 1. 选择阈值
    # threshold = np.percentile(data, 95)  # 选择95%分位数作为阈值

    # 2. 数据分割
    extremes = data[data > threshold] - threshold  # 只选择超过阈值的极端值

    # 3. 拟合概率分布
    params = genpareto.fit(extremes)  # 使用广义帕累托分布拟合极端值

    # 4. 参数估计
    shape, loc, scale = params  # GPD参数：形状参数、位置参数和尺度参数

    thresholdF = threshold + (scale / shape) * (pow((q * len(data) / len(extremes)), -shape) - 1)
    return thresholdF


def compute_anomaly_scoring(mux, x, time, WL):
    mux = mux.detach().numpy()
    x = x.detach().numpy()
    spt = np.mean((mux - x) ** 2, axis=2)
    print(spt.shape)
    sp = {}
    sw = {}

    for i in range(len(spt)):
        sw[time[i][WL - 1]] = np.mean(spt[i])
        for j in range(WL):
            if time[i][j] not in sp:
                sp[time[i][j]] = []
            sp[time[i][j]].append(spt[i][j])

    for key, value in sp.items():
        sp[key] = np.mean(value)

    return sp, sw


def loss_function(mux, varx, muz, logvarz, x):
    # 为重构概率，用于评价ˆx与x之间的相似度
    # 均方误差来计算对数似然
    # 对ˆx的均值作均方误差，然后对整个进行求均值值
    # recon_error = torch.sum((mux.reshape(-1, varx.shape[2]) - x.reshape(-1, varx.shape[2])) ** 2, dim=1)
    # reconstruction_loss = torch.mean(recon_error)

    # 生成正态分布再对其取对数
    # normal_dist = torch.distributions.Normal(loc=mux, scale=varx)
    # log_prob = normal_dist.log_prob(x)
    # reconstruction_loss = log_prob.sum(-1).mean()

    #直接均方误差
    # MSELoss = nn.MSELoss(reduction='sum')
    # reconstruction_loss = MSELoss(mux, x)

    #手写公式
    reconstruction_loss = torch.mean(torch.sum(torch.log(2 * varx) + (((mux - x) ** 2) / (2 * varx ** 2)), dim=2))

    # KL散度
    KL_divergence = -0.5 * torch.sum(1 + logvarz - torch.exp(logvarz) - muz ** 2)
    # print(reconstruction_loss, KL_divergence)

    return reconstruction_loss + KL_divergence
    # return -(reconstruction_loss - KL_divergence)


class GRU_VAE(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向GRU

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # utilize the GRU model in torch.nn
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2_mean = nn.Linear(256, 3)
        self.fc2_logvar = nn.Linear(256, 3)
        self.fc3 = nn.Linear(3, 256)
        self.fc4_mean = nn.Linear(256, output_size)
        self.fc4_var = nn.Linear(256, output_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2_mean(h1)), F.softplus(self.fc2_logvar(h1))

    def reparametrization(self, mu, logvar):
        # sigma = exp(0.5*log(sigma^2))= exp(log(sigma))
        std = torch.exp(0.5 * logvar)
        # N(mu, std^2) = N(0, 1) * std + mu

        # 从标准正态分布中选择于std.size()相同大小的值
        z = torch.randn(std.size()) * std + mu
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.relu(self.fc4_mean(h3)), F.softplus(self.fc4_var(h3))

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        h_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.gru(input_seq, (h_0.detach()))

        # 算出 均值和log方差
        # view()需要Tensor中的元素地址是连续的，但可能出现Tensor不连续的情况，所以先用 .contiguous() 将其在内存中变成连续
        # output.contiguous().view(-1, self.hidden_size)
        muz, logvarz = self.encode(output)
        # 重新采样， 因为直接对Z采用运算是不可导的，所以利用对正态分布采样在运算使其可导，Z = N~(0,1) * std + mu
        z = self.reparametrization(muz, logvarz)
        # 返回对z重新编码后算出均值与方差
        mux, varx = self.decode(z)
        return mux, varx, muz, logvarz


def createData(dataset, WL, WSS, IDs):
    # 初始化颜色
    color = ['red', 'black', 'blue', 'pink', 'purple', 'green', 'yellow', 'orange']
    # 初始单位
    units = ['km/h', 'V', 'A', '%', 'v', 'v', '℃', '℃']

    fig, axes = plt.subplots(8, 1, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.plot(dataset[:, 0], dataset[:, i + 1], color=color[i])
        ax.set_title(f"{IDs[i + 1]}")
        ax.set_xlabel('time')
        ax.set_ylabel(units[i])
        if i != 7: ax.get_xaxis().set_visible(False)  # 隐藏x坐标轴

    # subplots_adjust(self, left=None, bottom=None, right=None, top=None,wspace=None, hspace=None)
    # 参数1：left：指定子图左边缘距离
    # 参数2：bottom：指定子图底部距离
    # 参数3：right：指定子图右边缘距离
    # 参数4：top：指定子图顶部距离
    # 参数5：wspace：指定子图之间的横向距离
    # 参数6：hspace：指定子图之间的纵向距离
    # 所有的距离并不是指绝对数值而是一个相对比例，数值都在0-1之间。
    # left和bottom参数指的是左右两边的空白比例，
    # 例如left=0.5相当于左空白为0.5.而right和top参数则指得是整个图框所占的比例，例如top=0.9相当于上空白为0.1。
    fig.subplots_adjust(hspace=0.7)
    plt.show()

    # 对数据除时间之外的后8列进行运算
    # 最小-最大归一化
    # 防止除于0
    for i in range(1, dataset.shape[1]):
        if dataset[:, i].max(axis=0) - dataset[:, i].min(axis=0) != 0:
            dataset[:, i] = (dataset[:, i] - dataset[:, i].min(axis=0)) / (
                        dataset[:, i].max(axis=0) - dataset[:, i].min(axis=0))
        else:
            dataset[:, i] = 0

    # 对数据除时间之外的后8列转化类型
    dataset[:, 1:] = dataset[:, 1:].astype('float32')
    driving_data = []
    contiune_data = []
    data = []
    time = []
    lasttime = 0
    datediff = timedelta(seconds=10)
    for i in range(0, len(dataset)):
        if dataset[i][1] != 0:
            if len(contiune_data) == 0:
                lasttime = dataset[i][0]

            if dataset[i][0] - datediff <= lasttime:
                driving_data.append(dataset[i])
                lasttime = dataset[i][0]
            elif dataset[i][0] - datediff > lasttime:
                contiune_data.append(driving_data)
                driving_data.clear()
                driving_data.append(dataset[i])
                lasttime = dataset[i][0]

    # 放置最后一个连续时间段
    contiune_data.append(driving_data)
    contiune_data = np.array(contiune_data)

    for i in range(len(contiune_data)):
        for j in range(WL, len(contiune_data[i])):
            data.append(contiune_data[i][j - WL:j, 1:dataset.shape[1]])
            time.append(contiune_data[i][j - WL:j, 0])
            j += WSS

    return torch.tensor(np.array(data).astype('float32')), time


def data_description(WL, WSS):
    # parse_dates 表示将某一列设置为 时间类型
    # index_col=[0] 将[0]列作为索引值
    # low_memory=False 参数设置后，pandas会一次性读取csv中的所有数据，然后对字段的数据类型进行唯一的一次猜测。这样就不会导致同一字段的Mixed types问题了。
    data1 = pd.read_csv("data/datasets/No.1.csv", parse_dates=["yr_modahrmn"], low_memory=False)[:3000]
    data2 = pd.read_csv("data/datasets/No.2.csv", parse_dates=["yr_modahrmn"], low_memory=False)[:3000]
    data3 = pd.read_csv("data/datasets/No.3.csv", parse_dates=["yr_modahrmn"], low_memory=False)[:3000]
    data4 = pd.read_csv("data/datasets/No.4.csv", parse_dates=["yr_modahrmn"], low_memory=False)[:3000]
    print("data1 Shape-- ", data1.shape)
    print("data2 Shape-- ", data2.shape)
    print("data3 Shape-- ", data3.shape)
    print("data4 Shape-- ", data4.shape)

    data1 = data1.sort_values(by=['yr_modahrmn'], ascending=[True])
    data2 = data2.sort_values(by=['yr_modahrmn'], ascending=[True])
    data3 = data3.sort_values(by=['yr_modahrmn'], ascending=[True])
    data4 = data4.sort_values(by=['yr_modahrmn'], ascending=[True])

    IDs = ['yr_modahrmn', 'speed', 'total_volt', 'total_current', 'standard_soc', 'max_cell_volt', 'min_cell_volt',
           'max_temp',
           'min_temp']

    data1_tensor, time1 = createData(np.array(data1[IDs]), WL, WSS, IDs)
    data2_tensor, time2 = createData(np.array(data2[IDs]), WL, WSS, IDs)
    data3_tensor, time3 = createData(np.array(data3[IDs]), WL, WSS, IDs)
    data4_tensor, time4 = createData(np.array(data4[IDs]), WL, WSS, IDs)

    print("data1_tensor Shape-- ", data1_tensor.shape)
    print("data2_tensor Shape-- ", data2_tensor.shape)
    print("data3_tensor Shape-- ", data3_tensor.shape)
    print("data4_tensor Shape-- ", data4_tensor.shape)
    return data1_tensor, time1, data2_tensor, time2, data3_tensor, time3, data4_tensor, time4


device = torch.device("cpu")

if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

WL = 20
WSS = 10
data1_tensor, time1, data2_tensor, time2, data3_tensor, time3, data4_tensor, time4 = data_description(WL, WSS)

gru_vae = GRU_VAE(data1_tensor.shape[2], data1_tensor.shape[2], data1_tensor.shape[2])
optimizer = torch.optim.Adam(gru_vae.parameters(), lr=1e-3)

train_loader = DataLoader(data1_tensor, batch_size=128, shuffle=False)

prev_loss = 1000
max_epochs = 100

for epoch in range(max_epochs):
    allloss = 0
    for i, inputs in enumerate(train_loader):
        if i == 0: continue
        inputs = inputs.to(device)
        mux, varx, muz, logvarz = gru_vae(inputs)
        mux = mux.to(device)
        varx = varx.to(device)
        muz = muz.to(device)
        logvarz = logvarz.to(device)
        loss = loss_function(mux, varx, muz, logvarz, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        allloss += loss

    # if loss < prev_loss:
    #     torch.save(gru.state_dict(), 'gru_model.pt')  # save model parameters to files
    #     prev_loss = loss

    # if loss.item() < 1e-4:
    #     print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
    #     print("The loss value is reached")
    #     break

    if (epoch + 1) % 100 == 0:
        print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, allloss))

gru_vae = gru_vae.eval()  # switch to testing model

# ----------------- data2 -------------------

data2_tensor = data2_tensor.to(device)

mux, varx, muz, logvarz = gru_vae(data2_tensor)
mux = mux.to(device)
varx = varx.to(device)
muz = muz.to(device)
logvarz = logvarz.to(device)

loss = loss_function(mux, varx, muz, logvarz, data2_tensor)
print("test loss：", loss.item())

# ----------------- Anomaly scoring -------------------

sp2, sw2 = compute_anomaly_scoring(mux, data2_tensor, time2, WL)

sp_threshold = 0
for value in sp2.values():
    sp_threshold = max(sp_threshold, value)

threshold_hight = [sp_threshold for i in range(len(sp2))]

# ----------------- plot -------------------

fig = plt.figure(figsize=(12,4))
ax1 = fig.subplots()
ax2 = ax1.twinx()  # 使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
ax1.plot(sp2.keys(), sp2.values(), color='b', label='SP')
ax1.plot(sp2.keys(), threshold_hight, color='g', linestyle='--')
ax2.plot(sw2.keys(), sw2.values(), color='r', label='SW')

ax1.set_xlabel('time')
ax1.set_ylabel('SP')
ax2.set_ylabel('SW')
ax1.set_title("data2 SP SW in data1 model")
fig.legend()
plt.show()

# ----------------- data3 -------------------

data3_tensor = data3_tensor.to(device)

mux, varx, muz, logvarz = gru_vae(data3_tensor)
mux = mux.to(device)
varx = varx.to(device)
muz = muz.to(device)
logvarz = logvarz.to(device)

loss = loss_function(mux, varx, muz, logvarz, data3_tensor)
print("test loss：", loss.item())

# ----------------- Anomaly scoring -------------------

sp3, sw3 = compute_anomaly_scoring(mux, data3_tensor, time3, WL)

# ----------------- Threshold selection -------------------

sp_data = []
for value in sp3.values():
    sp_data.append(value)

sp_data = np.array(sp_data)

q = 1-1e-4
sp_thF = pot_analysis(sp_data, sp_threshold, q)
print(sp_thF)

# ----------------- plot -------------------

anomaly_point_time = []
anomaly_point_sp = []
thresholdF_hight = [sp_thF for i in range(len(sp3))]

for key, value in sp3.items():
    if value > sp_thF:
        anomaly_point_time.append(key)
        anomaly_point_sp.append(value)

fig = plt.figure(figsize=(12,4))
ax3 = fig.subplots()
ax3.plot(sp3.keys(), sp3.values(), color='b', label='SP')
# 画thF
ax3.plot(sp3.keys(), thresholdF_hight, c='g', linestyle='--')
ax3.scatter(anomaly_point_time, anomaly_point_sp, marker='o', s=40, c='r')

ax3.set_xlabel('time')
ax3.set_ylabel('SP')
ax3.set_title("data3 SP SW")
ax3.legend()

plt.show()

# ----------------- data4 -------------------

data4_tensor = data4_tensor.to(device)

mux, varx, muz, logvarz = gru_vae(data4_tensor)
mux = mux.to(device)
varx = varx.to(device)
muz = muz.to(device)
logvarz = logvarz.to(device)

loss = loss_function(mux, varx, muz, logvarz, data4_tensor)
print("test loss：", loss.item())

# ----------------- Anomaly scoring -------------------

sp4, sw4 = compute_anomaly_scoring(mux, data4_tensor, time4, WL)

# ----------------- Threshold selection -------------------

sp_data = []
for value in sp4.values():
    sp_data.append(value)

sp_data = np.array(sp_data)

q = 1-1e-4
sp_thF = pot_analysis(sp_data, sp_threshold, q)
print(sp_thF)

# ----------------- plot -------------------

anomaly_point_time = []
anomaly_point_sp = []
thresholdF_hight = [sp_thF for i in range(len(sp4))]

for key, value in sp4.items():
    if value > sp_thF:
        anomaly_point_time.append(key)
        anomaly_point_sp.append(value)

fig = plt.figure(figsize=(12,4))
ax4 = fig.subplots()
ax4.plot(sp4.keys(), sp4.values(), color='b', label='SP')
# 画thF
ax4.plot(sp4.keys(), thresholdF_hight, c='g', linestyle='--')
ax4.scatter(anomaly_point_time, anomaly_point_sp, marker='o', s=40, c='r')

ax4.set_xlabel('time')
ax4.set_ylabel('SP')
ax4.set_title("data4 SP SW")
ax4.legend()

plt.show()