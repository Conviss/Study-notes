import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

def random_selection(X, Y, w, b):
    misclassification = []
    for i in range(X.shape[0]):
        if (np.dot(w, X[i]) + b) * Y[i] <= 0:
            misclassification.append(i)
    if len(misclassification) == 0: return -1
    else:
        return misclassification[np.random.randint(0, len(misclassification))]

def compute_gradient(x, y):
    n = len(x)
    dw = np.zeros((n,))
    db = y
    for i in range(n):
        dw[i] = x[i] * y
    return dw, db

def gradient_descent(X, Y, w, b, lambda_):
    count = 0
    while True:
        i = random_selection(X, Y, w, b)
        if i == -1: break  # no misclassification points
        dw, db = compute_gradient(X[i], Y[i])
        w = w + lambda_ * dw
        b = b + lambda_ * db
        print(count, f"x{i:d}", w, b, f"{w[0]:.0f}x(1) + {w[1]:.0f}x(2) + {b:.0f}")
        count += 1
    return w, b

X = np.array([(3, 3), (4, 3), (1, 1)])
Y = np.array([1, 1, -1])
w = np.zeros((X.shape[1],))
b = 0
lambda_ = 1
final_w, final_b = gradient_descent(X, Y, w, b, lambda_)
print(final_w, final_b)

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