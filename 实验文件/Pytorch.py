import torch
import numpy as np
#%%
torch.cuda.is_available()

#%%
# 创建一个 5*3 的矩阵
x = torch.empty(5, 3)
print(x)

#%%
# 创建一个随机初始化的 5*3 矩阵
rand_x = torch.rand(5, 3)
print(rand_x)

#%%
# 创建一个数值皆是 0，类型为 long 的矩阵
zero_x = torch.zeros(5, 3, dtype=torch.long)
print(zero_x)

#%%
# tensor 数值是 [5.5, 3]
tensor1 = torch.tensor([5.5, 3])
print(tensor1)

#%%
# 显示定义新的尺寸是 5*3，数值类型是 torch.double
tensor2 = tensor1.new_ones(5, 3, dtype=torch.double)  # new_* 方法需要输入 tensor 大小
print(tensor2)

#%%
# 修改数值类型
tensor3 = torch.randn_like(tensor2, dtype=torch.float)
print('tensor3: ', tensor3)

#%%
print(tensor3.size())
# 输出: torch.Size([5, 3])

#%%
tensor4 = torch.rand(5, 3)
print('tensor3 + tensor4= ', tensor3 + tensor4)
print('tensor3 + tensor4= ', torch.add(tensor3, tensor4))
# 新声明一个 tensor 变量保存加法操作的结果
result = torch.empty(5, 3)
torch.add(tensor3, tensor4, out=result)
print('add result= ', result)
# 直接修改变量
# 可以改变 tensor 变量的操作都带有一个后缀 _, 例如 x.copy_(y), x.t_() 都可以改变 x 变量
tensor3.add_(tensor4)
print('tensor3= ', tensor3)

#%%
# 访问 tensor3 第一列数据
print(tensor3[:, 0])

#%%
# 对 Tensor 的尺寸修改
x = torch.randn(4, 4)
y = x.view(16)
# -1 表示除给定维度外的其余维度的乘积
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

#%%
# 如果 tensor 仅有一个元素，可以采用 .item() 来获取类似 Python 中整数类型的数值：
x = torch.randn(1)
print(x)
print(x.item())

#%%
# 实现 Tensor 转换为 Numpy 数组的例子如下所示，调用 tensor.numpy() 可以实现这个转换操作。
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

#%%
# 两者是共享同个内存空间的
a.add_(1)
print(a)
print(b)

#%%
# 转换的操作是调用 torch.from_numpy(numpy_array) 方法
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

#%%
# 当 CUDA 可用的时候，可用运行下方这段代码，采用 torch.device() 方法来改变 tensors 是否在 GPU 上进行计算操作
if torch.cuda.is_available():
    device = torch.device("cuda")          # 定义一个 CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 显示创建在 GPU 上的一个 tensor
    x = x.to(device)                       # 也可以采用 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # .to() 方法也可以改变数值类型

