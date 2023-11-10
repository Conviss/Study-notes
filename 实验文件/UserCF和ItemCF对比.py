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

#划分数据集
def SplitData(data, M, k, seed):
    test = dict()
    train = dict()
    random.seed(seed)
    for index, row in data.iterrows():
        userId = row['userId']
        movieId = row['movieId']

        if random.randint(0,M) == k:
            if userId not in test:
                test[userId] = set()
            test[userId].add(movieId)
        else:
            if userId not in train:
                train[userId] = set()
            train[userId].add(movieId)
    return train, test

#计算用户相似度
def UserSimilarity(train):
    item_users = dict()     #物品-用户表
    for user, items in train.items():
        for item in items:
            if item not in item_users:
                item_users[item] = set()
            item_users[item].add(user)

    C = dict()
    for i, users in item_users.items():
        for u in users:
            for v in users:
                if u == v:
                    continue
                if u not in C:
                    C[u] = dict()
                if v not in C[u]:
                    C[u][v] = 0
                C[u][v] += 1

    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            if u not in W:
                W[u] = dict()
            W[u][v] = cuv / math.sqrt(len(train[u]) * len(train[v]) * 1.0)

    return W

def GetRecommendationUserCF(user, train, W, K, N):
    rank = dict()
    interacted_items = train[user]
    for v, wuv in sorted(W[user].items(), key=lambda x:x[1], reverse=True)[:K]:
        for i in train[v]:
            if i in interacted_items:
               continue
            if i not in rank:
               rank[i] = 0
            rank[i] += wuv
    rank = dict(sorted(rank.items(), key=lambda x:x[1], reverse=True)[:N])
    return rank

#计算用户相似度
def ItemSimilarity(train):
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items:
            if i not in N:
                N[i] = 0
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                if i not in C:
                    C[i] = dict()
                if j not in C[i]:
                    C[i][j] = 0
                C[i][j] += 1

    W = dict()
    for i, related_items in C.items():
        for j, cij in related_items.items():
            if i not in W:
                W[i] = dict()
            W[i][j] = cij / math.sqrt(N[i] * N[j] * 1.0)

    return W

def GetRecommendationItemCF(user, train, W, K, N):
    rank = dict()
    interacted_items = train[user]
    for i in interacted_items:
        for j, wij in sorted(W[i].items(), key=lambda x: x[1], reverse=True)[:K]:
            if j in interacted_items:
                continue
            if j not in rank:
                rank[j] = 0
            rank[j] += wij
    rank = dict(sorted(rank.items(), key=lambda x:x[1], reverse=True)[:N])
    return rank

def Recall(train, test, W, K, N, GetRecommendation):
    hit = 0
    all = 0
    for user in train.keys():
        if user not in test:
            continue
        tu = test[user]
        rank = GetRecommendation(user, train, W, K, N)
        for item, pui in rank.items():
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)

def Precision(train, test, W, K, N, GetRecommendation):
    hit = 0
    all = 0
    for user in train.keys():
        if user not in test:
            continue
        tu = test[user]
        rank = GetRecommendation(user, train, W, K, N)
        for item, pui in rank.items():
            if item in tu:
                hit += 1
        all += len(rank)
    return hit / (all * 1.0)

def Coverage(train, test, W, K, N, GetRecommendation):
    recommend_items = set()
    all_items = set()
    for user, items in train.items():
        for item in items:
            all_items.add(item)
            rank = GetRecommendation(user, train, W, K, N)
            for item, pui in rank.items():
                recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)

def Popularity(train, test, W, K, N, GetRecommendation):
    item_popularity = dict()
    for user, items in train.items():
        for item in items:
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = GetRecommendation(user, train, W, K, N)
        for item, pui in rank.items():
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret

#读取数据
Movies_Ratings = pd.read_table('data/ml-1m/ratings.dat', header=None, delimiter='::', engine='python')[:1000]
Movies_Ratings.columns = ['userId','movieId','rating','timestamp']
#打印数据
print(Movies_Ratings)
#划分数据集，令M=5，k=0，seed=3
train, test = SplitData(Movies_Ratings, 5, 0, 3)
#打印训练集
# print(train)
#打印测试集
# print(test)

W_UserCF = UserSimilarity(train)
W_ItemCF = ItemSimilarity(train)

X = []
Y_recall_UserCF = []
# Y_precision_UserCF = []
Y_coverage_UserCF = []
Y_popularity_UserCF = []
Y_recall_ItemCF = []
# Y_precision_ItemCF = []
Y_coverage_ItemCF = []
Y_popularity_ItemCF = []
K = 5
while K <= 160:
    X.append(K)
    Y_recall_UserCF.append(Recall(train, test, W_UserCF, K, 80, GetRecommendationUserCF))
    # Y_precision_UserCF.append(Precision(train, test, W_UserCF, K, 80, GetRecommendationUserCF))
    Y_coverage_UserCF.append(Coverage(train, test, W_UserCF, K, 80, GetRecommendationUserCF))
    Y_popularity_UserCF.append(Popularity(train, test, W_UserCF, K, 80, GetRecommendationUserCF))
    Y_recall_ItemCF.append(Recall(train, test, W_ItemCF, K, 80, GetRecommendationItemCF))
    # Y_precision_ItemCF.append(Precision(train, test, W_ItemCF, K, 80, GetRecommendationItemCF))
    Y_coverage_ItemCF.append(Coverage(train, test, W_ItemCF, K, 80, GetRecommendationItemCF))
    Y_popularity_ItemCF.append(Popularity(train, test, W_ItemCF, K, 80, GetRecommendationItemCF))
    K *= 2

# UserCF和ItemCF算法在不同K值下的召回率曲线
plt.plot(X, Y_recall_UserCF, label = 'UserCF', color = 'b', marker='d')
plt.plot(X, Y_recall_ItemCF, label = 'ItemCF', color = 'r', marker='s')
plt.title("UserCF和ItemCF算法在不同K值下的召回率曲线")
plt.ylabel('召回率')
plt.xlabel('K值')
plt.legend()
plt.show()

# UserCF和ItemCF算法在不同K值下的覆盖率曲线
plt.plot(X, Y_coverage_UserCF, label = 'UserCF', color = 'b', marker='d')
plt.plot(X, Y_coverage_ItemCF, label = 'ItemCF', color = 'r', marker='s')
plt.title("UserCF和ItemCF算法在不同K值下的覆盖率曲线")
plt.ylabel('覆盖率')
plt.xlabel('K值')
plt.legend()
plt.show()

#UserCF和ItemCF算法在不同K值下的流行度曲线
plt.plot(X, Y_popularity_UserCF, label = 'UserCF', color = 'b', marker='d')
plt.plot(X, Y_popularity_ItemCF, label = 'ItemCF', color = 'r', marker='s')
plt.title("UserCF和ItemCF算法在不同K值下的流行度曲线")
plt.ylabel('流行度')
plt.xlabel('K值')
plt.legend()
plt.show()
