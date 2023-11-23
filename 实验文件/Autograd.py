import torch
#%%
# 张量
# 开始创建一个 tensor， 并让 requires_grad=True 来追踪该变量相关的计算操作：
x = torch.ones(2, 2, requires_grad=True)
print(x)

#%%
y = x + 2
print(y)

#%%
print(y.grad_fn)

#%%
z = y * y * 3
out = z.mean() #求均值

print('z=', z)
print('out=', out)

#%%
# 一个 Tensor 变量的默认 requires_grad 是 False
# 可以像上述定义一个变量时候指定该属性是 True
# 也可以定义变量后，调用 .requires_grad_(True) 设置为 True
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

#%%
# 梯度
# 接下来就是开始计算梯度，进行反向传播的操作
# out 变量是一个标量，因此 out.backward() 相当于 out.backward(torch.tensor(1.))
out.backward()
# 输出梯度 d(out)/dx
print(x.grad)

#%%
# 一般来说，torch.autograd 就是用于计算雅克比向量(vector-Jacobian)乘积的工具
x = torch.randn(3, requires_grad=True)

y = x * 2
# y.data.norm() 张量y每个元素进行平方，然后对它们求和，最后取平方根。 这些操作计算就是所谓的L2或欧几里德范数 。
while y.data.norm() < 1000:
    y = y * 2

print(y)

#%%
# y 不再是一个标量，torch.autograd
# 不能直接计算完整的雅克比行列式，但我们可以通过简单的传递向量给 backward() 方法作为参数得到雅克比向量的乘积
# 就等于 y 与 v 的向量的乘积之后就变成一个数，然后再返回求导
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# grad属性在反向传播过程中是累加的，每一次反向传播梯度都会累加之前的梯度。因此每次重新计算梯度前都要将梯度清零
a.grad.data.zero_()

#%%
# 最后，加上 with torch.no_grad() 就可以停止追踪变量历史进行自动梯度计算：
# 在外面会加上自动梯度路径，在里面运算不会加上
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
