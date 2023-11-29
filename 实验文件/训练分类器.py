# 在训练分类器前，当然需要考虑数据的问题。通常在处理如图片、文本、语音或者视频数据的时候，
# 一般都采用标准的 Python 库将其加载并转成 Numpy 数组，然后再转回为 PyTorch 的张量。
# 对于图像，可以采用 Pillow, OpenCV 库；
# 对于语音，有 scipy 和 librosa;
# 对于文本，可以选择原生 Python 或者 Cython 进行加载数据，或者使用 NLTK 和 SpaCy。

# 训练流程如下：
#
# 通过调用 torchvision 加载和归一化 CIFAR10 训练集和测试集；
# 构建一个卷积神经网络；
# 定义一个损失函数；
# 在训练集上训练网络；
# 在测试集上测试网络性能。
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

show = ToPILImage()  # 可以把Tensor转成Image，方便可视化
import torchvision.datasets as dsets

batch_size = 100
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化 先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
])

# Cifar110 dataset
train_dataset = dsets.CIFAR10(root='./data/classImg',
                              train=True,
                              download=True,
                              transform=transform
                              )
test_dataset = dsets.CIFAR10(root='./data/classImg',
                             train=False,
                             download=True,
                             transform=transform
                             )
# 加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True
                                           )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True
                                          )
#%%
# 测试集大小
print(train_dataset[0])

#%%
# 展示图片的函数
import matplotlib.pyplot as plt
fig = plt.figure()
classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    (_, label) = train_dataset[i]
    plt.imshow(train_loader.dataset.data[i],cmap=plt.cm.binary)
    plt.title("Labels: {}".format(classes[label]))
    plt.xticks([])
    plt.yticks([])
plt.show()

#%%
import torch.nn as nn
import torch

input_size = 3072  # 3*32*32
hidden_size1 = 500  # 第一次隐藏层个数
hidden_size2 = 200  # 第二次隐藏层个数
num_classes = 10  # 分类个数
num_epochs = 5  # 批次次数
batch_size = 100  # 批次大小
learning_rate = 1e-3


# 定义两层神经网络
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)  # 输入
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)  # 两层隐藏层计算
        self.layer3 = nn.Linear(hidden_size2, num_classes)  # 输出

    def forward(self, x):
        out = torch.relu(self.layer1(x))  # 隐藏层1
        out = torch.relu(self.layer2(out))  # 隐藏层2
        out = self.layer3(out)
        return out


net = Net(input_size, hidden_size1, hidden_size2, num_classes)
#%%
import torch.optim as optim
# 这里采用类别交叉熵函数和带有动量的 SGD 优化方法：
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

#%%
# 训练网络-Net
import time
start = time.time()
batch_size = 1000  # 批次大小
for epoch in range(2):
    print('current epoch + %d' % epoch)
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader, 0):
        print(images.size())
        print(labels.size())
        images = images.view(images.size(0), -1)
        labels = torch.tensor(labels, dtype=torch.long)
        # 梯度清零
        optimizer.zero_grad()
        outputs = net(images)  # 将数据集传入网络做前向计算
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 0:  # 每1000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
print('Finished Training')
print('Finished Training! Total cost time: ', time.time()-start)
#%%
# 模型准确率
# prediction
total = 0
correct = 0
acc_list_test = []
for images, labels in test_loader:
    images = images.view(images.size(0), -1)
    outputs = net(images)  # 将数据集传入网络做前向计算

    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()
    acc_list_test.append(100 * correct / total)

print('Accuracy = %.2f' % (100 * correct / total))
plt.plot(acc_list_test)
plt.xlabel('Epoch')
plt.ylabel('Accuracy On TestSet')
plt.show()