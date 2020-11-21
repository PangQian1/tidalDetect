#!/usr/bin/python3
import csv
import random
import warnings

import pandas as pd
from prettytable import PrettyTable
from sklearn import linear_model, svm, neighbors, naive_bayes, tree, ensemble
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from xgboost import XGBClassifier
import numpy as np
from Tidal import dao

random.seed(123)
warnings.filterwarnings('ignore')

#加载数据
def load_data(sample_file, label_file, test_file):
    """
    从文件中读取数据
    :param sample_file: 数据信息
    :param label_file: 标签信息
    :return: xs,ys
    """
    xs = pd.read_csv(sample_file, header=None)
    ys = pd.read_csv(label_file, header=None)
    test = pd.read_csv(test_file, header=None)
    return xs, ys, test

#计算评价指标
def get_metrics(y_pred, y_true):
    """
    计算评价指标
    :param y_pred: 预测值
    :param y_true: 真实值
    :return: acc, f1, p, recall, p_r
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    p_r = precision_recall_curve(y_true, y_pred)
    return acc, f1, p, recall, p_r

#模型训练与预测
def get_result(name, model, x_train, y_train, x_test, y_test, test, table):
    """
    模型训练与预测
    :param name: 模型名
    :param model: 模型
    :param x_train: 训练集数据x
    :param y_train: 训练集标签y
    :param x_test: 测试集数据x
    :param y_test: 测试集标签y
    :param table: 结果记录表格
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    dao.score(model.predict(test))
    acc, f1, p, recall, p_r = get_metrics(y_pred, y_test)
    table.add_row([name, f'{acc * 100:.4f}', f'{f1 * 100:.4f}', f'{p * 100:.4f}', f'{recall * 100:.4f}'])

    preGrid(model)

#划分数据集
def split_data(xs, ys, ratio=0.7):
    """
    划分数据集
    :param xs: 原始数据
    :param ys: 原始标签
    :param ratio: 训练集比例
    :return: x_train, y_train, x_test, y_test
    """
    # 建立数据索引
    indexes_1, indexes_0 = list(range(int(len(xs) * 0.5))), list(range(int(len(xs) * 0.5), len(xs)))
    # 打乱顺序
    random.shuffle(indexes_1), random.shuffle(indexes_0)
    # 数据划分为train和test
    indexes_train = indexes_1[:int(len(indexes_1) * ratio)] + indexes_0[:int(len(indexes_0) * ratio)]
    indexes_test = indexes_1[int(len(indexes_1) * ratio):] + indexes_0[int(len(indexes_0) * ratio):]
    x_train, y_train = xs.loc[indexes_train, :], ys.loc[indexes_train, :]
    x_test, y_test = xs.loc[indexes_test, :], ys.loc[indexes_test, :]
    return x_train, y_train, x_test, y_test

def main():
    """
    机器学习方法对比
    """
    # 读取数据
    xs, ys, test = load_data('E:\\G-1149\\trafficCongestion\\训练数据\\4小时文件\\trainData\\res\\resSam.csv',
                             'E:\\G-1149\\trafficCongestion\\训练数据\\4小时文件\\trainData\\res\\resLabel.csv',
                             #'C:\\Users\\98259\\Desktop\\6.9学习相关文档\\样本数据\\fiftMin\\samplePeakHour_训练数据 - 副本Line.csv',
                             'data/test_4.csv'
                             )
    x_train, y_train, x_test, y_test = split_data(xs, ys, ratio=0.7)
    #print(np.isnan(test).any())
    test = test.loc[:,:]

    # 测试不同方法
    # 评价指标
    table = PrettyTable(['Methods', 'Accuracy', 'F1', 'Precision', 'Recall'])
    methods = {}

    # 2. Support Vector Machines
    svc = svm.SVC()
    methods['Support Vector Machines'] = svc

    # 4. Nearest Neighbors
    knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
    methods['Nearest Neighbors'] = knn

    # 7.4 Gradient Tree Boosting
    gbdt = ensemble.GradientBoostingClassifier()
    methods['Gradient Tree Boosting'] = gbdt

    # 7.5 XGBoosting
    #xg = XGBClassifier()
    #methods['XGBoosting'] = xg

    for name, model in methods.items():
        get_result(name, model, x_train, y_train, x_test, y_test, test, table)
    print(table)

def preGrid(model):
    gridLinkPeerPath = 'E:/G-1149/trafficCongestion/网格化/gridLinkPeer_14.csv'
    gridTidalPath = 'E:/G-1149/trafficCongestion/网格化/tidal/gridTidal_ml_14.csv'
    linkStatusPath = "E:/G-1149/trafficCongestion/网格化/linkStatus_14_完整.csv"

    statusDict = {}
    with open(linkStatusPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            statusDict[r[0]] = r[1:]

    f = open(gridTidalPath, 'w', encoding='utf-8', newline='')  # newline解决空行问题
    csv_writer = csv.writer(f)

    with open(gridLinkPeerPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            if (len(r) == 1):
                continue
            linkPeer = r[1:]
            for i in range(len(linkPeer)):
                if (i % 2 != 0):
                    pre = np.array(linkStatus + statusDict[linkPeer[i]]).astype(float)
                    pre = pre.reshape(-1, 32)
                    pre = model.predict(pre)
                    csv_writer.writerow([r[0], pre[0]])
                else:
                    linkStatus = statusDict[linkPeer[i]]

    f.close()

from sklearn.metrics import accuracy_score
if __name__ == '__main__':
    main()


