import numpy as np
#%%  列表推导式与条件赋值
def my_func(x):
    return 2*x
# [* for i in *] 。其中，第一个 * 为映射函数，其输入为后面 i 指代的内容，第二个 * 表示迭代的对象。
[my_func(i) for i in range(5)]

#%% 多层嵌套
[m+'_'+n for m in ['a', 'b'] for n in ['c', 'd']]

#%% 条件赋值
value = 'cat' if 2>1 else 'dog'

#%%
L = [1, 2, 3, 4, 5, 6, 7, 8, 9]
[i if i <= 5 else 5 for i in L]

#%%  匿名函数与 map 方法
my_func = lambda x: 2*x
my_func(3)

#%%
[(lambda x: 2*x)(i) for i in range(5)]

#%% map 函数 列表推导式的匿名函数映射
list(map(lambda x: 2*x, range(5)))

#%% 多个输入值的函数映射，可以通过追加迭代对象实现：
list(map(lambda x, y: str(x)+'_'+y, range(5), list('abcde')))

#%%   zip 对象与 enumerate 方法
L1, L2, L3 = list('abc'), list('def'), list('hij')

#%% 压缩
list(zip(L1, L2, L3))
#%%
# 元组是有序且不可更改的集合。在Python中，元组使用圆括号 () 编写的。
tuple(zip(L1, L2, L3))

#%% 循环迭代
for i, j, k in zip(L1, L2, L3):
    print(i, j, k)

#%% enumerate 是一种特殊的打包，它可以在迭代时绑定迭代元素的遍历序号
L = list('abcd')
for index, value in enumerate(L):
    print(index, value)

#%% 用 zip 对象也能够简单地实现这个功能
for index, value in zip(range(len(L)), L):
    print(index, value)

print(list(zip(range(len(L)), L)))
#%% 两个列表建立字典映射
dict(zip(L1, L2))
#%% 压缩
zipped = list(zip(L1, L2, L3))
zipped
#%% 解压缩
list(zip(*zipped))
#%% np 数组的构造
np.array([1,2,3])
#%% 等差序列：np.linspace, np.arange
np.linspace(1,5,11) # 起始、终止（包含）、样本个数
#%%
np.arange(1,5,2) # 起始、终止（不包含）、步长

#%% 特殊矩阵：zeros, eye, full
np.zeros((2,3)) # 传入元组表示各维度大小
#%%
np.eye(3) # 3*3 的单位矩阵
#%%
np.eye(3, k=1) # 偏移主对角线 1 个单位的伪单位矩阵
#%%
np.full((2,3), 10) # 元组传入大小，10 表示填充数值
#%%
np.full((2,3), [1,2,3]) # 通过传入列表填充每列的值
#%% 随机矩阵：np.random
# 最常用的随机生成函数为 rand, randn, randint, choice ，它们分别表示 0-1 均匀分布的随机数组、标准正态的随机数组、随机整数组和随机列表抽样：
np.random.rand(3) # 生成服从 0-1 均匀分布的三个随机数
#%%
np.random.rand(3, 3) # 注意这里传入的不是元组，每个维度大小分开输入
#%% 对于服从区间 a 到 b 上的均匀分布可以如下生成：
a, b = 5, 15
(b - a) * np.random.rand(3) + a

#%% randn 生成了 N(0, I) 的标准正态分布：
np.random.randn(3)
#%%
np.random.randn(2, 2)
#%% 对于服从方差为 σ^2 均值为 µ 的一元正态分布可以如下生成：
sigma, mu = 2.5, 3
mu + np.random.randn(3) * sigma
#%% randint 可以指定生成随机整数的最小值最大值和维度大小：
low, high, size = 5, 15, (2,2)
np.random.randint(low, high, size)
#%% choice 可以从给定的列表中，以一定概率和方式抽取结果，当不指定概率时为均匀采样，默认抽取方式为有放回抽样：
my_list = ['a', 'b', 'c', 'd']
np.random.choice(my_list, 2, replace=False, p=[0.1, 0.7, 0.1 ,0.1]) #replace = False, 无放回抽样，= True 有放回抽样
#%%
np.random.choice(my_list, (3,3))
#%% 当返回的元素个数与原列表相同时，等价于使用 permutation 函数，即打散原列表：
np.random.permutation(my_list)
#%%
np.random.seed(0)
#%%
np.random.rand()

#%% 转置：T
np.zeros((2,3)).T
#%% 合并操作：r_, c_ 对于二维数组而言，r_ 和 c_ 分别表示上下合并和左右合并：
np.r_[np.zeros((2,3)),np.zeros((2,3))]
#%%
np.c_[np.zeros((2,3)),np.zeros((2,3))]

#%% 一维数组和二维数组进行合并时，应当把其视作列向量，在长度匹配的情况下只能够使用左右合并的 c_ 操作：
try:
    np.r_[np.array([1,2]),np.zeros((2,1))]
except Exception as e:
    Err_Msg = e
#%%
Err_Msg
#%%
np.r_[np.array([1,2,3]),np.zeros(3)]
#%%
np.c_[np.array([1,2]),np.zeros((2,3))]
#%% 维度变换：reshape 能够帮助用户把原数组按照新的维度重新排列。在使用时有两种模式，分别为 C 模式和 F 模式，分别以逐行和逐列的顺序进行填充读取。
target = np.arange(8).reshape(2,4)
target
#%%
target.reshape((4,2), order='C')  # 按照行读取和填充
#%%
target.reshape((4,2), order='F') # 按照列读取和填充
#%% 特别地，由于被调用数组的大小是确定的，reshape 允许有一个维度存在空缺，此时只需填充-1 即可：
target.reshape((4,-1))

#%% 下面将 n*1 大小的数组转为 1 维数组的操作是经常使用的：
target = np.ones((3,1))
target
#%%
target.reshape(-1)
#%% 数组的切片模式支持使用 slice 类型的 start:end:step 切片，还可以直接传入列表指定某个维度的索引进行切片：
target = np.arange(9).reshape(3,3)
target
#%%
target[:-1, [0,2]] #逗号前是行，逗号后是列，前面为从0到-1， 后面为选择0，2列
#%% 还可以利用 np.ix_ 在对应的维度上使用布尔索引，但此时不能使用 slice 切片：
target[np.ix_([True, False, True], [True, False, True])]
#%%
target[np.ix_([1,2], [True, False, True])]
#%% 当数组维度为 1 维时，可以直接进行布尔索引，而无需 np.ix_ ：
new = target.reshape(-1)
new[new%2==0]

#%% where 是一种条件函数，可以指定满足条件与不满足条件位置对应的填充值：
a = np.array([-1,1,-1,0])
np.where(a>0, a, 5) # 对应位置为 True 时填充 a 对应元素，否则填充 5

#%% 这三个函数返回的都是索引，nonzero 返回非零数的索引，argmax, argmin 分别返回最大和最小数的索引：
a = np.array([-2,-5,0,1,3,-1])
#%%
np.nonzero(a)
#%%
a.argmax()
#%%
a.argmin()

#%% any 指当序列至少 存在一个 True 或非零元素时返回 True ，否则返回 False, all 指当序列元素 全为 True 或非零元素时返回 True ，否则返回 False
a = np.array([0,1])
#%%
a.any()
#%%
a.all()

#%% cumprod, cumsum 分别表示累乘和累加函数，返回同长度的数组，diff 表示和前一个元素做差，由于第一个元素为缺失值，因此在默认参数情况下，返回长度是原数组减 1
a = np.array([1,2,3,4])
#%%
a.cumprod()
#%%
a.cumsum()
#%%
np.diff(a)

#%% 常用的统计函数包括 max, min, mean, median, std, var, sum, quantile ，其中分位数计算是全局方法，因此不能通过 array.quantile 的方法调用：
target = np.arange(5)
target
#%%
target.max()
#%%
np.quantile(target, 0.5) # 0.5 分位数  将数组按从小到大进行排序之后，对应的分位点位置的值

#%% 是对于含有缺失值的数组，它们返回的结果也是缺失值，如果需要略过缺失值，必须使用 nan* 类型的函数，上述的几个统计函数都有对应的 nan* 函数。
target = np.array([1, 2, np.nan])
target
#%%
target.max()
#%%
np.nanmax(target)
#%%
np.nanquantile(target, 0.5)

#%% 对于协方差和相关系数分别可以利用 cov, corrcoef 如下计算：
target1 = np.array([1, 3, 5, 9])
target2 = np.array([1,5,3,-9])
#%%
np.cov(target1, target2)
#%%
np.corrcoef(target1, target2)

#%% 二维 Numpy 数组中统计函数的 axis 参数，它能够进行某一个维度下的统计特征计算，当 axis=0 时结果为列的统计指标，当 axis=1 时结果为行的统计指标：
target = np.arange(1,10).reshape(3,-1)
target
#%%
target.sum(0)
#%%
target.sum(1)

#%% 广播机制
# 当一个标量和数组进行运算时，标量会自动把大小扩充为数组大小，之后进行逐元素操作：
res = 3 * np.ones((2,2)) + 1
res
#%%
res = 1 / res
res
#%% 当两个数组维度完全一致时，使用对应元素的操作，否则会报错，除非其中的某个数组的维度是 m × 1 或者
# 1 × n ，那么会扩充其具有 1 的维度为另一个数组对应维度的大小。例如，1 × 2 数组和 3 × 2 数组做逐元素
# 运算时会把第一个数组扩充为 3 × 2 ，扩充时的对应数值进行赋值。但是，需要注意的是，如果第一个数组
# 的维度是 1 × 3 ，那么由于在第二维上的大小不匹配且不为 1 ，此时报错。
res = np.ones((3,2))
res
#%%
res * np.array([[2,3]]) # 扩充第一维度为 3
#%%
res * np.array([[2],[3],[4]]) # 扩充第二维度为 2
#%%
res * np.array([[2]]) # 等价于两次扩充
#%% 当一维数组 Ak 与二维数组 Bm,n 操作时，等价于把一维数组视作 A1,k 的二维数组，使用的广播法则与【b】中一致，当 k! = n 且 k, n 都不是 1 时报错。
np.ones(3) + np.ones((2,3))
#%%
np.ones(3) + np.ones((2,1))
#%%
np.ones(1) + np.ones((2,3))
#%% 向量与矩阵的计算 向量内积：dot
a = np.array([1,2,3])
b = np.array([1,3,5])
a.dot(b)
#%% 向量范数和矩阵范数：np.linalg.norm
martix_target = np.arange(4).reshape(-1,2)
martix_target
#%%
np.linalg.norm(martix_target, 'fro')
#%%
np.linalg.norm(martix_target, np.inf)
#%%
np.linalg.norm(martix_target, 2)
#%%
vector_target = np.arange(4)
vector_target
#%%
np.linalg.norm(vector_target, np.inf)
#%%
np.linalg.norm(vector_target, 2)
#%%
np.linalg.norm(vector_target, 3)
#%% 矩阵乘法
a = np.arange(4).reshape(-1,2)
a
#%%
b = np.arange(-4,0).reshape(-1,2)
b
#%%
a@b
#%% 练习1 利用列表推导式写矩阵乘法
M1 = np.random.rand(2,3)
M2 = np.random.rand(3,4)
res = np.empty((M1.shape[0],M2.shape[1]))
for i in range(M1.shape[0]):
    for j in range(M2.shape[1]):
        item = 0
        for k in range(M1.shape[1]):
            item += M1[i][k] * M2[k][j]
            res[i][j] = item

((M1@M2 - res) < 1e-15).all() # 排除数值误差
#%%
res = [[sum(M1[i][k] * M2[k][j] for k in range(M1.shape[1])) for j in range(M2.shape[1])] for i in range(M1.shape[0])]
((M1@M2 - res) < 1e-15).all() # 排除数值误差
#%% 更新矩阵
A = np.arange(1, 10).reshape(3,3)
B = [[A[i][j] * sum(1/A[i,:]) for j in range(A.shape[1])] for i in range(A.shape[0])]
B