'''
@author: Frank
@contact: 734778368@qq.com
@file: model.py
@time: 2021-04-04 15:23
@desc:
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
import pickle
from itertools import product
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as sm

from sklearn.ensemble import RandomForestRegressor
import re

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

print('读入数据')
data = pd.read_csv(r'C://Users//Frank//Desktop//predict_future_sales//data.csv')
print('数据读入成功。。。')
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

X_train = data[data.date_block_num < 33].drop(['Unnamed0','item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['Unnamed0','item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['Unnamed0','item_cnt_month'], axis=1)

print(X_train.shape,X_train.columns)
print('--------------------------------------------------')
print(X_train.head(8))
print(Y_train.shape)
print(X_valid.shape)
print(Y_valid.shape)
print('--------------------------------------------------')
print(X_test.shape,X_test.columns)
print(X_test.head(8))

del data
gc.collect();

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

ts = time.time()
train_data = lgb.Dataset(data=X_train, label=Y_train)
valid_data = lgb.Dataset(data=X_valid, label=Y_valid)
#
print(train_data)
time.time() - ts

params = {"objective": "regression", "metric": "rmse", 'n_estimators': 10000, 'early_stopping_rounds': 50,
          "num_leaves": 200, "learning_rate": 0.01, "bagging_fraction": 0.9,
          "feature_fraction": 0.3, "bagging_seed": 0}

lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000)
#gbm.save_model('model.txt')
from sklearn.externals import joblib
joblib.dump(lgb_model, 'data_model.pkl')

# Y_pred=lgb_model.predict(X_valid,num_iteration=lgb_model.best_iteration)


Y_test = lgb_model.predict(X_valid).clip(0, 20)

print('The rmse of prediction is:', mean_squared_error(Y_valid,Y_test) ** 0.5)

# Y_test.to_csv(r'C://Users//Frank//Desktop//predict_future_sales//test_res.csv')

print(Y_test)

