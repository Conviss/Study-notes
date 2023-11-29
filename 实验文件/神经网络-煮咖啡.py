import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


# 随机生成数据
def load_coffee_data():
    rng = np.random.default_rng(2) # 创建一个种子为2的生成器，可以为空，空时会随机分配一个种子。
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5  # 12-15 min is best
    X[:, 0] = X[:, 0] * (285 - 150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3 / (260 - 175) * t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1, 1))

X,Y = load_coffee_data();
print(X.shape, Y.shape)

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")

#计算X的归一化
mean_X = np.mean(X)
std_X = np.std(X)
Xn = (X - mean_X) / std_X

print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

Xt = np.tile(Xn,(1000,1)) #np.tile(a, (x, y)) 表示将a矩阵按照行复制成x个，按照列复制成y个， 先复制完行或列再复制列或行
Yt = np.tile(Y,(1000,1))

# 对精度，np返回float64，F。Linear是float32
Xt = torch.tensor(Xt).to(torch.float32)
Yt = torch.tensor(Yt).to(torch.float32)


train = [[Xt[i], Yt[i]] for i in range(Xt.shape[0])]

train_loader = DataLoader(
    dataset=train,              # 数据，封装进Data.TensorDataset()类的数据
    batch_size=200,             # 每块的大小
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多进程（multiprocess）来读数据
)

print(len(train), len(train_loader))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(2, 3)
        self.layer2 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.sigmoid(self.layer1(x))
        x = F.sigmoid(self.layer2(x))
        return x

net = Net()

print(net)
#输出模型中的权重
params = list(net.parameters())

k=0
for i in params:
    l = 1
    print("该层的结构："+str(list(i.size())))
    print(i)
    for j in i.size():
        l *= j
    print( "参数和："+str(l))
    k = k+l

print( "总参数和："+ str(k))


#该语句定义一个损失函数并指定编译优化。model.compile
criterion = nn.BCELoss() #二分类交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=0.01)

for epoch in range(100):
    print('current epoch + %d' % epoch)
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # 梯度清零
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print('Finished Training')

#更新后的权重
params = list(net.parameters())

for i in params:
    l = 1
    print("该层的结构："+str(list(i.size())))
    print(i)

#创建测试数据并对其规范化
X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example

X_testn = (X_test - mean_X) / std_X

X_testn = torch.tensor(X_testn).to(torch.float32)
print(X_testn)

predictions = net(X_testn)
print("predictions = \n", predictions)

predictions = predictions.detach().numpy()
#将概率转换为决策：
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")

# pro = Y == 1
# neg = Y == 0
#
# X_temp = X[:,0].reshape(-1, 1)
# X_time = X[:,1].reshape(-1, 1)
#
# fig, ax = plt.subplots(1, 1) #创建一个1*1的子图
# ax.scatter(X_temp[pro], X_time[pro], marker='x', s=40, c='r', label="Y = 1")
# ax.scatter(X_temp[neg], X_time[neg], marker='o', s=40, c='b', label="Y = 0")
# ax.scatter(X_test[:,0], X_test[:,1], marker='+', s=40, c='g', label="test")
# ax.legend( fontsize='xx-large')
# ax.set_ylabel('time', fontsize='xx-large')
# ax.set_xlabel('temp', fontsize='xx-large')
# plt.show()