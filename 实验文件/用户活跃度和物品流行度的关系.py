import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

#读取数据
Movies_Ratings = pd.read_table('data/ml-1m/ratings.dat', header=None, delimiter='::', engine='python')
Movies_Ratings.columns = ['userId','movieId','rating','timestamp']
#打印数据
print(Movies_Ratings)

item_popularity = dict()    #物品流行度
user_activity = dict()      #用户活跃度

user_item = dict()          #用户-物品
activity_item = dict()        #被活跃度为K的用户评分过的物品的数量
activity_popularity = dict()        #被活跃度为K的用户评分过的物品的平均流行度

item_popularity_data = []   #物品流行度数据
item_number_data = []       #流行度为K的物品数量数据
user_activity_data = []     #用户活跃度数据
user_number_data = []       #活跃度为K的用户数量数据

#统计物品的流行度以及用户的活跃度
for data, row in Movies_Ratings.iterrows():
    userId = row['userId']
    movieId = row['movieId']

    if userId not in user_activity:
        user_activity[userId] = 0
    if movieId not in item_popularity:
        item_popularity[movieId] = 0
    if userId not in user_item:
        user_item[userId] = []

    user_activity[userId] += 1
    item_popularity[movieId] += 1
    user_item[userId].append(movieId)

#统计被活跃度为K的用户评分过的物品
for userId, activity in user_activity.items():
    if activity not in activity_item:
        activity_item[activity] = 0
    if activity not in activity_popularity:
        activity_popularity[activity] = 0

    for item in user_item[userId]:
        activity_item[activity] += 1
        activity_popularity[activity] += item_popularity[item]

#统计被活跃度为K的用户评分过的物品的平均流行度
for activity in activity_popularity.keys():
    activity_popularity[activity] /= activity_item[activity] * 1.0

#做用户活跃度和物品流行度的关系图
plt.scatter(activity_popularity.keys(), activity_popularity.values())
plt.title("用户活跃度和物品流行度")
plt.ylabel('平均物品热门度')
plt.xlabel('用户活跃度')
plt.show()

