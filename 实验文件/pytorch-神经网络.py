import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 在 PyTorch 中 torch.nn 专门用于实现神经网络。
# 其中 nn.Module 包含了网络层的搭建，以及一个方法-- forward(input) ，并返回网络的输出 outptu .

# 对于神经网络来说，一个标准的训练流程是这样的：
# 定义一个多层的神经网络
# 对数据集的预处理并准备作为网络的输入
# 将数据输入到网络
# 计算网络的损失
# 反向传播，计算梯度
# 更新网络的梯度，一个简单的更新规则是 weight = weight - learning_rate * gradient

# 首先定义一个神经网络，下面是一个 5 层的卷积神经网络，包含两层卷积层和三层全连接层：
class Net(nn.Module):
    #定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(Net, self).__init__() #复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(1, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
        self.conv2 = nn.Conv2d(6, 16, 5)# 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.fc1   = nn.Linear(16*5*5, 120) # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2   = nn.Linear(120, 84)#定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3   = nn.Linear(84, 10)#定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。

    #定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    # 这里必须实现 forward 函数，而 backward 函数在采用 autograd 时就自动定义好了，在 forward 方法可以采用任何的张量操作。
    def forward(self, x):
        # max-pooling 采用一个 (2,2) 的滑动窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #输入x经过卷积conv1之后，经过激活函数ReLU（原来这个词是激活函数的意思），使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        # 核(kernel)大小是方形的话，可仅定义一个数字，如 (2,2) 用 2 即可
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = x.view(-1, self.num_flat_features(x)) #view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x)) #输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x)) #输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.fc3(x) #输入x经过全连接3，然后更新x
        return x

    #使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
    def num_flat_features(self, x):
        size = x.size()[1:] # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()

#%%
# 以下代码是为了看一下我们需要训练的参数的数量
print(net)
params = list(net.parameters())

k=0
for i in params:
    l =1
    print("该层的结构："+str(list(i.size())))
    for j in i.size():
        l *= j
    print( "参数和："+str(l))
    k = k+l

print( "总参数和："+ str(k))

# net.parameters() 可以返回网络的训练参数，使用例子如下：
params = list(net.parameters())
print('参数数量: ', len(params))
# conv1.weight
print('第一个参数大小: ', params[0].size())

#%%
# 随机定义一个变量输入网络
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

#%%
# 清空所有参数的梯度缓存，然后计算随机梯度进行反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))

# torch.nn 只支持小批量(mini-batches)数据，也就是输入不能是单个样本，
# 比如对于 nn.Conv2d 接收的输入是一个 4 维张量--nSamples * nChannels * Height * Width 。
# 所以，如果你输入的是单个样本，需要采用 input.unsqueeze(0) 来扩充一个假的 batch 维度，即从 3 维变为 4 维。

#%%
output = net(input)
# 定义伪标签
target = torch.randn(10)
# 调整大小，使得和 output 一样的 size
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

#%%
# 清空所有参数的梯度缓存
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

#%%
# 简单实现权重的更新例子
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
# 但是这只是最简单的规则，深度学习有很多的优化算法，不仅仅是 SGD，
# 还有 Nesterov-SGD, Adam, RMSProp 等等，为了采用这些不同的方法，
# 这里采用 torch.optim 库，使用例子如下所示：
#%%
# 创建优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练过程中执行下列操作
optimizer.zero_grad() # 清空梯度缓存
output = net(input)
loss = criterion(output, target)
loss.backward()
# 更新权重
optimizer.step()