# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
共享单车使用量预测
"""
import numpy as np
import sklearn.ensemble as se
import sklearn.metrics as sm
import sklearn.utils as su
import matplotlib.pyplot as mp

# 加载数据集
data = []
with open('./bike_day.csv', 'r') as f:
    for line in f.readlines():
        data.append(line[:-1].split(','))
data = np.array(data)
# 划分测试集与训练集
header = data[0, 2:13]
x = data[1:, 2:13].astype('f8')
y = data[1:, -1].astype('f8')

# 打乱数据集
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:]

# 训练模型:随机森林
model = se.RandomForestRegressor(
    max_depth=10, n_estimators=1000,
    min_samples_split=2)
model.fit(train_x, train_y)

# 评估模型
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

fi = model.feature_importances_
mp.figure('Feature Importance', facecolor='lightgray')
mp.subplot(211)
mp.title('bike day', fontsize=16)
mp.ylabel('Importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
x = np.arange(fi.size)
sorted_inds = fi.argsort()[::-1]
mp.bar(x, fi[sorted_inds], 0.8, color='dodgerblue',
       label='day')
mp.xticks(x, header[sorted_inds])

mp.tight_layout()
mp.legend()

# 加载数据集
data = []
with open('./bike_hour.csv', 'r') as f:
    for line in f.readlines():
        data.append(line[:-1].split(','))
data = np.array(data)

header = data[0, 2:14]
x = data[1:, 2:14].astype('f8')
y = data[1:, -1].astype('f8')


x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:]

# 训练模型
model = se.RandomForestRegressor(
    max_depth=10, n_estimators=1000,
    min_samples_split=2)
model.fit(train_x, train_y)

# 评估模型
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
fi = model.feature_importances_

mp.subplot(212)
mp.title('bike hour', fontsize=16)
mp.ylabel('Importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
x = np.arange(fi.size)
sorted_inds = fi.argsort()[::-1]
mp.bar(x, fi[sorted_inds], 0.8,
       color='dodgerblue',
       label='day')
mp.xticks(x, header[sorted_inds])
mp.legend()
mp.show()
