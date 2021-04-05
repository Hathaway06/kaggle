from sklearn.externals import joblib
import re
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error



data = pd.read_csv(r'C://Users//Frank//Desktop//predict_future_sales//data.csv')
print('数据读入成功。。。')
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

X_valid = data[data.date_block_num == 33].drop(['Unnamed0','item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']

lgb_model=joblib.load('data_model.pkl')

Y_test = lgb_model.predict(X_valid).clip(0, 20)

print('The rmse of prediction is:', mean_squared_error(Y_valid,Y_test) ** 0.5)

# Y_test.to_csv(r'C://Users//Frank//Desktop//predict_future_sales//test_res.csv')

print(len(Y_test))
print(X_valid.shape)
X_valid['item_cnt_month'] = Y_test
print(X_valid.loc[:,['item_id','shop_id','item_cnt_month']].head(8))


test=pd.read_csv(r'C://Users//Frank//Desktop//predict_future_sales//test.csv')

good_sales = test.merge(X_valid, on=['item_id','shop_id'], how='left')
res_sales=good_sales.loc[:,['ID','item_cnt_month']]

res_sales['item_cnt_month'][res_sales['item_cnt_month'].isnull()]=0

res_sales.to_csv('submission.csv',index=False)

