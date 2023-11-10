import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

#读取数据
# Book_Ratings = pd.read_csv('data/Book reviews/BX-Book-Ratings.csv', sep=';', encoding='utf-8')
Book_Ratings = pd.read_table('data/ml-1m/ratings.dat', header=None, delimiter='::', engine='python')
Book_Ratings.columns = ['User-ID','ISBN','rating','timestamp']
#打印数据
print(Book_Ratings)

item_popularity = dict()    #物品流行度
user_activity = dict()      #用户活跃度

item_number = dict()        #流行度为K的物品数量
user_number = dict()        #活跃度为K的用户数量

item_popularity_data = []   #物品流行度数据
item_number_data = []       #流行度为K的物品数量数据
user_activity_data = []     #用户活跃度数据
user_number_data = []       #活跃度为K的用户数量数据

#统计物品的流行度以及用户的活跃度
for data, row in Book_Ratings.iterrows():
    user_id = row['User-ID']
    isbn = row['ISBN']

    if user_id not in user_activity:
        user_activity[user_id] = 0
    if isbn not in item_popularity:
        item_popularity[isbn] = 0

    user_activity[user_id] += 1
    item_popularity[isbn] += 1

#统计用户活跃度为K的数量
for activity in user_activity.values():
    if activity not in user_number:
        user_number[activity] = 0
    user_number[activity] += 1

#统计物品流行度为K的数量
for popularity in item_popularity.values():
    if popularity not in item_number:
        item_number[popularity] = 0
    item_number[popularity] += 1

#对数据进行排序
item_number = dict(sorted(item_number.items()))
user_number = dict(sorted(user_number.items()))

#数据转存至数组
for key, value in item_number.items():
    item_popularity_data.append(key)
    item_number_data.append(value)

for key, value in user_number.items():
    user_activity_data.append(key)
    user_number_data.append(value)

#做物品长尾分布图
plt.plot(item_popularity_data, item_number_data)
plt.title("物品长尾分布")
plt.ylabel('流行度为K的物品数量')
plt.xlabel('流行度')
plt.show()

plt.scatter(np.log(item_popularity_data), np.log(item_number_data))
plt.title("物品长尾分布")
plt.ylabel('流行度为K的物品数量')
plt.xlabel('流行度')
plt.show()

#做用户长尾分布图
plt.plot(user_activity_data, user_number_data)
plt.title("用户长尾分布")
plt.ylabel('活跃度为K的用户数量')
plt.xlabel('活跃度')
plt.show()

plt.scatter(np.log(user_activity_data), np.log(user_number_data))
plt.title("用户长尾分布")
plt.ylabel('活跃度为K的用户数量')
plt.xlabel('活跃度')
plt.show()
