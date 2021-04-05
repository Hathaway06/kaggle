#导入包
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import time
import gc
import pickle
from itertools import product

#读入数据
    # 训练集有六列，分别介绍日期，月份，商店，商品，价格和日销量
    # 测试集有三列，分别是ID，商店，和商品。

sales_train = pd.read_csv(r'C://Users//Frank//Desktop//predict_future_sales//sales_train.csv')
test = pd.read_csv(r'C://Users//Frank//Desktop//predict_future_sales//test.csv')

# print(sales_train.head(10))

# print('how many lines in train set:', sales_train.shape)
# print('unique items in train set:', sales_train['item_id'].nunique())
# print('unique shops in train set:', sales_train['shop_id'].nunique())
# print('how many lines in test set:', test.shape)
# print('unique items in test set:', test['item_id'].nunique())
# print('unique shops in test set:', test['shop_id'].nunique())

#基线模型预测
    # 训练集中的数据是 商品-商店-每天的销售。而要求预测的是商品-商店-每月的销售，因此需要合理使用groupby()和agg()函数。
    # 训练集没有出现过的 商品-商店组合，一律填零，最终的结果需要限幅在 [0,20]区间。

sales_train_subset = sales_train[sales_train['date_block_num'] == 33]
# print(sales_train_subset.head())

grouped = sales_train_subset[['shop_id','item_id','item_cnt_day']].groupby(['shop_id','item_id']).agg({'item_cnt_day':'sum'}).reset_index()
grouped = grouped.rename(columns={'item_cnt_day' : 'item_cnt_month'})
# print(grouped.head())

test = pd.merge(test,grouped, on = ['shop_id','item_id'], how = 'left')
# print(test.head())
test['item_cnt_month'] = test['item_cnt_month'].fillna(0).clip(0,20)#限制结果范围在0-20
# print(test.head())
test = test[['ID','item_cnt_month']]
submission = test.set_index('ID')
submission.to_csv('submission_baseline.csv')

#节省存储空间
def downcast_dtypes(df):
    cols_float64 = [c for c in df if df[c].dtype == 'float64']
    cols_int64_32 = [c for c in df if df[c].dtype in ['int64', 'int32']]
    df[cols_float64] = df[cols_float64].astype(np.float32)
    df[cols_int64_32] = df[cols_int64_32].astype(np.int16)
    return df
sales_train = downcast_dtypes(sales_train)
test = downcast_dtypes(test)
# print(sales_train.info())

#数据探索（每件商品销量、每个商店销量、每类商品销量、销量和价格离群值）
#每件商品的销量（得到每件商品每月的销量，并且把产品id设置为index）
sales_by_item_id = sales_train.pivot_table(index=['item_id'],values=['item_cnt_day'],
                                        columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)##有点没看懂
sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_item_id.columns.values[0] = 'item_id'

# print(sales_by_item_id.tail())
#月销量图
# sales_by_item_id.sum()[1:].plot(legend=True, label="Monthly sum")

#分析有多少商品在最近的连续六个月内，没有销量(没有历史数据的商品和商店全部置为0)
    # 训练集一共21807种商品，其中有12391种在最近的六个月没有销量。
    # 测试集一共5100种商品，其中有164种在训练中最近六个月没有销量，共出现了164 * 42 = 6888次。
    # Tips：在最终的预测结果中，我们可以将这些商品的销量大胆地设置为零。
outdated_items = sales_by_item_id[sales_by_item_id.loc[:,'27':].sum(axis=1)==0]
# print('Outdated items:', len(outdated_items))
test = pd.read_csv(r'C://Users//Frank//Desktop//predict_future_sales//test.csv')
# print('unique items in test set:', test['item_id'].nunique())
# print('Outdated items in test set:', test[test['item_id'].isin(outdated_items['item_id'])]['item_id'].nunique())
    # Outdated items: 12391（训练集无效商品个数）
    # unique items in test set: 5100
    # Outdated items in test set: 164（测试集中无效商品个数）


#删除重复行
print("duplicated lines in sales_train is", len(sales_train[sales_train.duplicated()]))


#每个商店的销量
    # 共有 60 个商店，坐落在31个城市,城市的信息可以作为商店的一个特征。
    # 这里先分析下哪些商店是最近才开的，哪些是已经关闭了的，同样分析最后六个月的数据。
    #
    # shop_id = 36 是新商店
    # shop_id = [0 1 8 11 13 17 23 29 30 32 33 40 43 54] 可以认为是已经关闭了。
    # Tips：新商店，可以直接用第33个月来预测34个月的销量，因为它没有任何历史数据。而已经关闭的商店，销量可以直接置零

sales_by_shop_id = sales_train.pivot_table(index=['shop_id'],values=['item_cnt_day'],
                                        columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_shop_id.columns = sales_by_shop_id.columns.droplevel().map(str)
sales_by_shop_id = sales_by_shop_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_shop_id.columns.values[0] = 'shop_id'

for i in range(27,34):
    print('Not exists in month',i,sales_by_shop_id['shop_id'][sales_by_shop_id.loc[:,'0':str(i)].sum(axis=1)==0].unique())

for i in range(27,34):
    print('Shop is outdated for month',i,sales_by_shop_id['shop_id'][sales_by_shop_id.loc[:,str(i):].sum(axis=1)==0].unique())

#引入类别
item_categories = pd.read_csv(r'C://Users//Frank//Desktop//predict_future_sales//items.csv')
item_categories = item_categories[['item_id','item_category_id']]

# print(item_categories.head())

#商品与类别结合
sales_train_merge_cat = pd.merge(sales_train,item_categories, on = 'item_id', how = 'left')
sales_train_merge_cat.head()

#查找离群值
# plt.figure(figsize=(10,4))
# plt.xlim(-100,3000)
# sns.boxplot(x = sales_train['item_cnt_day'])
# print('Sale volume outliers:',sales_train['item_cnt_day'][sales_train['item_cnt_day']>1001].unique())
# plt.figure(figsize=(10,4))
# plt.xlim(-10000,320000)
# sns.boxplot(x = sales_train['item_price'])
# print('Sale price outliers:',sales_train['item_price'][sales_train['item_price']>300000].unique())
# # plt.show()
#
sales_train = sales_train[sales_train['item_cnt_day'] <1001]
sales_train = sales_train[sales_train['item_price'] < 300000]
# plt.figure(figsize=(10,4))
# plt.xlim(-100,3000)
# sns.boxplot(x = sales_train['item_cnt_day'])
#
# plt.figure(figsize=(10,4))
# plt.xlim(-10000,320000)
# sns.boxplot(x = sales_train['item_price'])


#将离群值置为中位数
median = sales_train[(sales_train['date_block_num'] == 4) & (sales_train['shop_id'] == 32)\
                     & (sales_train['item_id'] == 2973) & (sales_train['item_price']>0)].item_price.median()
sales_train.loc[sales_train['item_price']<0,'item_price'] = median
print(median)


#测试集分析
    # 测试集有5100 种商品，42个商店。刚好就是5100 * 42 = 214200种 商品-商店组合。可以分为三大类
    #
    # 363种商品在训练集没有出现，363*42=15,246种商品-商店没有数据，约占7%。
    # 87550种商品-商店组合是只出现过商品，没出现过组合。约占42%。
    # 111404种商品-商店组合是在训练集中完整出现过的。约占51%。

test = pd.read_csv(r'C://Users//Frank//Desktop//predict_future_sales//test.csv')
good_sales = test.merge(sales_train, on=['item_id','shop_id'], how='left').dropna()
good_pairs = test[test['ID'].isin(good_sales['ID'])]
no_data_items = test[~(test['item_id'].isin(sales_train['item_id']))]

print('1. Number of good pairs:', len(good_pairs))
print('2. No Data Items:', len(no_data_items))
print('3. Only Item_id Info:', len(test)-len(no_data_items)-len(good_pairs))

#店铺信息
shops = pd.read_csv(r'C://Users//Frank//Desktop//predict_future_sales//shops.csv')
print(shops.head())

sales12 = np.array(sales_by_shop_id.loc[sales_by_shop_id['shop_id'] == 12 ].values)
sales12 = sales12[:,1:].reshape(-1)
sales55 = np.array(sales_by_shop_id.loc[sales_by_shop_id['shop_id'] == 55 ].values)
sales55 = sales55[:,1:].reshape(-1)
months = np.array(sales_by_shop_id.loc[sales_by_shop_id['shop_id'] == 12 ].columns[1:])
np.corrcoef(sales12,sales55)

#商店异常值转换
sales_train.loc[sales_train['shop_id'] == 0,'shop_id'] = 57
sales_train.loc[sales_train['shop_id'] == 1,'shop_id'] = 58
sales_train.loc[sales_train['shop_id'] == 11,'shop_id'] = 10
sales_train.loc[sales_train['shop_id'] == 40,'shop_id'] = 39


#商店信息编码###依据是什么
shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+','').str.strip()
shops['shop_city'] = shops['shop_name'].str.partition(' ')[0]#表示店铺名第一位是城市名称
shops['shop_type'] = shops['shop_name'].apply(lambda x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'NO_DATA')
# print(shops.head())


shops['shop_city_code'] = LabelEncoder().fit_transform(shops['shop_city'])
shops['shop_type_code'] = LabelEncoder().fit_transform(shops['shop_type'])
# print(shops.head())

categories = pd.read_csv(r'C://Users//Frank//Desktop//predict_future_sales//item_categories.csv')

lines1 = [26,27,28,29,30,31]
lines2 = [81,82]
for index in lines1:
    category_name = categories.loc[index,'item_category_name']
#    print(category_name)
    category_name = category_name.replace('Игры','Игры -')
#    print(category_name)
    categories.loc[index,'item_category_name'] = category_name
for index in lines2:
    category_name = categories.loc[index,'item_category_name']
#    print(category_name)
    category_name = category_name.replace('Чистые','Чистые -')
#    print(category_name)
    categories.loc[index,'item_category_name'] = category_name
category_name = categories.loc[32,'item_category_name']
#print(category_name)
category_name = category_name.replace('Карты оплаты','Карты оплаты -')
#print(category_name)
categories.loc[32,'item_category_name'] = category_name


categories['split'] = categories['item_category_name'].str.split('-')
categories['type'] = categories['split'].map(lambda x:x[0].strip())
categories['subtype'] = categories['split'].map(lambda x:x[1].strip() if len(x)>1 else x[0].strip())
categories = categories[['item_category_id','type','subtype']]
categories.head()

categories['cat_type_code'] = LabelEncoder().fit_transform(categories['type'])
categories['cat_subtype_code'] = LabelEncoder().fit_transform(categories['subtype'])
categories.head()

#首先将训练集中的数据统计好月销量
ts = time.time()
matrix = []
cols = ['date_block_num', 'shop_id', 'item_id']
for i in range(34):
    sales = sales_train[sales_train.date_block_num == i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols, inplace=True)
time.time() - ts

sales_train['revenue'] = sales_train['item_price'] * sales_train['item_cnt_day']

groupby = sales_train.groupby(['item_id', 'shop_id', 'date_block_num']).agg({'item_cnt_day': 'sum'})
groupby.columns = ['item_cnt_month']
groupby.reset_index(inplace=True)
matrix = matrix.merge(groupby, on=['item_id', 'shop_id', 'date_block_num'], how='left')
matrix['item_cnt_month'] = matrix['item_cnt_month'].fillna(0).clip(0, 20).astype(np.float16)
matrix.head()

test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)


cols = ['date_block_num', 'shop_id', 'item_id']
matrix = pd.concat([matrix, test[['item_id', 'shop_id', 'date_block_num']]], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True)  # 34 month
print(matrix.head())



#将上面得到的商店，商品类别等信息与矩阵融合起来
ts = time.time()
items = pd.read_csv(r'C://Users//Frank//Desktop//predict_future_sales//items.csv')
matrix = matrix.merge(items[['item_id','item_category_id']], on = ['item_id'], how = 'left')
matrix = matrix.merge(categories[['item_category_id','cat_type_code','cat_subtype_code']], on = ['item_category_id'], how = 'left')
matrix = matrix.merge(shops[['shop_id','shop_city_code','shop_type_code']], on = ['shop_id'], how = 'left')
matrix['shop_city_code'] = matrix['shop_city_code'].astype(np.int8)
matrix['shop_type_code'] = matrix['shop_type_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['cat_type_code'] = matrix['cat_type_code'].astype(np.int8)
matrix['cat_subtype_code'] = matrix['cat_subtype_code'].astype(np.int8)
time.time() - ts

#lag operation产生延迟信息，可以选择延迟的月数
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

#月销量（每个商品-商店）的历史信息
ts = time.time()
matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')
time.time() - ts


#月销量（所有商品-商店）均值的历史信息

ts = time.time()
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_avg_item_cnt')
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


#月销量（每件商品）均值和历史特征
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


#月销量（每个商店）均值和历史特征
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

#月销量（每个商品类别）均值和历史特征
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_cat_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_cat_avg_item_cnt')
matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

#月销量（商品类别-商店）均值和历史特征
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_category_id','shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_cat_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_category_id','shop_id'], how='left')
matrix['date_cat_shop_avg_item_cnt'] = matrix['date_cat_shop_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_cat_shop_avg_item_cnt')
matrix.drop(['date_cat_shop_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


#月销量（商品类别_大类）均值和历史特征
ts = time.time()
group = matrix.groupby(['date_block_num', 'cat_type_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_type_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','cat_type_code'], how='left')
matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_type_avg_item_cnt')
matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


#月销量（商品-商品类别_大类）均值和历史特征
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_id','cat_type_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_type_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id','cat_type_code'], how='left')
matrix['date_item_type_avg_item_cnt'] = matrix['date_item_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_type_avg_item_cnt')
matrix.drop(['date_item_type_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts


#月销量（商店_城市）均值和历史特征
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_city_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_city_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num',  'shop_city_code'], how='left')
matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_city_avg_item_cnt')
matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

#月销量（商品-商店_城市）均值和历史特征
ts = time.time()
group = matrix.groupby(['date_block_num','item_id', 'shop_city_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_item_city_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'shop_city_code'], how='left')
matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_city_avg_item_cnt')
matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

#趋势特征，半年来价格的变化
ts = time.time()
group = sales_train.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['item_id'], how='left')
matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

group = sales_train.groupby(['date_block_num', 'item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

lags = [1, 2, 3, 4, 5, 6, 12]
matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

for i in lags:
    matrix['delta_price_lag_' + str(i)] = \
        (matrix['date_item_avg_item_price_lag_' + str(i)] - matrix['item_avg_item_price']) / matrix[
            'item_avg_item_price']


def select_trend(row):
    for i in lags:
        if row['delta_price_lag_' + str(i)]:
            return row['delta_price_lag_' + str(i)]
    return 0


matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
matrix['delta_price_lag'].fillna(0, inplace=True)

# https://stackoverflow.com/questions/31828240/first-non-null-value-per-row-from-a-list-of-pandas-columns/31828559
# matrix['price_trend'] = matrix[['delta_price_lag_1','delta_price_lag_2','delta_price_lag_3']].bfill(axis=1).iloc[:, 0]
# Invalid dtype for backfill_2d [float16]

fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
    fetures_to_drop += ['date_item_avg_item_price_lag_' + str(i)]
    fetures_to_drop += ['delta_price_lag_' + str(i)]


#为什么要删掉价格特征啊？？？
matrix.drop(fetures_to_drop, axis=1, inplace=True)

time.time() - ts


#每个月天数
matrix['month'] = matrix['date_block_num'] % 12
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)

#每种商品首次和最后一次卖出的销量
ts = time.time()
cache = {}
matrix['item_shop_last_sale'] = -1
matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():
    key = str(row.item_id)+' '+str(row.shop_id)
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
        cache[key] = row.date_block_num
time.time() - ts

#Months since the first sale for each shop/item pair and for item only.
ts = time.time()
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
time.time() - ts


#因为使用了12个月作为延迟特征，必然由大量的数据是NA值，将最开始11个月的原始特征删除，并且对于NA值我们需要把它填充为0。
ts = time.time()
matrix = matrix[matrix.date_block_num > 11]
time.time() - ts

ts = time.time()
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)
    return df

matrix = fill_na(matrix)
time.time() - ts
matrix.to_csv(r'C://Users//Frank//Desktop//predict_future_sales//data.csv')
print(matrix)

# #建模
# # def plot_features(booster, figsize):
# #     fig, ax = plt.subplots(1,1,figsize=figsize)
# #     return plot_importance(booster=booster, ax=ax)
#
# data = pd.read_pickle('data_simple.pkl')
#
# X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
# Y_train = data[data.date_block_num < 33]['item_cnt_month']
# X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
# Y_valid = data[data.date_block_num == 33]['item_cnt_month']
# X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
#
#
# del data
# gc.collect();
#
# import lightgbm as lgb
#
# ts = time.time()
# train_data = lgb.Dataset(data=X_train, label=Y_train)
# valid_data = lgb.Dataset(data=X_valid, label=Y_valid)
#
# time.time() - ts
#
# params = {"objective": "regression", "metric": "rmse", 'n_estimators': 10000, 'early_stopping_rounds': 50,
#           "num_leaves": 200, "learning_rate": 0.01, "bagging_fraction": 0.9,
#           "feature_fraction": 0.3, "bagging_seed": 0}
#
# lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000)
# Y_test = lgb_model.predict(X_test).clip(0, 20)
#
#

