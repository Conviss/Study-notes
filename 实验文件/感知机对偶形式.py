import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

def random_selection(X, Y, a, b, G):
    misclassification = []
    for i in range(X.shape[0]):
        res = 0
        for j in range(X.shape[0]):
            res += G[i][j] * Y[j] * a[j]
        if Y[i] * (res + b) <= 0:
            misclassification.append(i)
    if len(misclassification) == 0: return -1
    else:
        return misclassification[np.random.randint(0, len(misclassification))]

def gradient_descent(X, Y, lambda_, a, b):
    G = [[np.dot(X[i], X[j]) for j in range(X.shape[0])] for i in range(X.shape[0])]
    count = 0
    print(count, a, b)
    while True:
        i = random_selection(X, Y, a, b, G)
        if i == -1: break  # no misclassification points
        a[i] += lambda_
        b += lambda_ * Y[i]
        count += 1
        print(count, a, b, f"x{i:d}")
    return a, b

X = np.array([(3, 3), (4, 3), (1, 1)])
Y = np.array([1, 1, -1])
lambda_ = 1

while True:
    a = np.zeros((X.shape[0],))
    b = 0
    final_a, final_b = gradient_descent(X, Y, lambda_, a, b)
    print(final_a, final_b)

    final_w = np.zeros((X.shape[1],))
    for i in range(X.shape[0]):
        final_w += a[i] * Y[i] * X[i]
    print(final_w)
    if 0 not in final_w: break

pro = Y == 1
neg = Y == -1
x1 = np.arange(0, 5)
x2 = - (final_b + (x1 * final_w[0])) / final_w[1]

plt.scatter(X[pro][:,0], X[pro][:,1], marker='o', c='b') #marker = 'o' 表示点用o表示
plt.scatter(X[neg][:,0], X[neg][:,1], marker='x', c='b') #marker = 'x' 表示点用x表示
plt.plot(x1, x2, c='r')
plt.title("感知机")
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()