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

def GetRecommendation(user, train, W, K, N):
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

def Recall(train, test, W, K, N):
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

def Precision(train, test, W, K, N):
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

def Coverage(train, test, W, K, N):
    recommend_items = set()
    all_items = set()
    for user, items in train.items():
        for item in items:
            all_items.add(item)
            rank = GetRecommendation(user, train, W, K, N)
            for item, pui in rank.items():
                recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)

def Popularity(train, test, W, K, N):
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
Movies_Ratings = pd.read_table('data/ml-1m/ratings.dat', header=None, delimiter='::', engine='python')[:29415]
Movies_Ratings.columns = ['userId','movieId','rating','timestamp']
#打印数据
print(Movies_Ratings)
#划分数据集，令M=5，k=0，seed=3
train, test = SplitData(Movies_Ratings, 5, 0, 3)
#打印训练集
# print(train)
#打印测试集
# print(test)

W = UserSimilarity(train)

print("K 准确率 召回率 覆盖率 流行度")
K = 5
while K <= 160:
    recall = Recall(train, test, W, K, 80)
    precision = Precision(train, test, W, K, 80)
    coverage = Coverage(train, test, W, K, 80)
    popularity = Popularity(train, test, W, K, 80)
    print('%d %.2f%% %.2f%% %.2f%% %.6f' % (K, precision * 100, recall * 100, coverage * 100, popularity))
    K *= 2
