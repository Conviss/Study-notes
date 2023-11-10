import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

X = [1, 2, 3, 4, 5]
Y1 = [2, 3, 4, 5, 6]
Y2 = [4, 6, 1, 3, 6]

#UserCF和ItemCF算法在不同K值下的流行度曲线
plt.plot(X, Y1, label = 'UserCF', color = 'b', marker='d')
plt.plot(X, Y2, label = 'ItemCF', color = 'r', marker='s')
plt.title("UserCF和ItemCF算法在不同K值下的流行度曲线")
plt.ylabel('流行度')
plt.xlabel('K值')
plt.legend()
plt.show()