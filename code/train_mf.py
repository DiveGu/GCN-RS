import numpy as np
import pandas as pd

def get_init_data(path,col,sep):
    data=pd.read_csv(path,sep = sep,encoding='latin-1',header=None)
    data.columns=col
    return data

user_data_path='F:/data/ml-100k/u.user'
user_data_col=['userID','age','gender','occupation','zipCode']

item_data_path='F:/data/ml-100k/u.item'
item_data_col=['itemID', 'movieTitle', 'releaseDate', 'videoReleaseDate',
              'IMDB_URL' ,'unknown', 'Action', 'Adventure', 'Animation',
              'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western']

rate_data_path='F:/data/ml-100k/u.data'
rate_data_col=['userID', 'itemID', 'rating', 'timestamp']

train_data_path='F:/data/ml-100k/u1.base'
test_data_path='F:/data/ml-100k/u1.test'

#读取原始的 用户数据 物品数据 交互数据
user_data=get_init_data(user_data_path,user_data_col,'|')
#print(user_data.head())

item_data=get_init_data(item_data_path,item_data_col,'|')
#print(item_data.head())

rate_data=get_init_data(rate_data_path,rate_data_col,'\t')
train_data=get_init_data(train_data_path,rate_data_col,'\t')
test_data=get_init_data(test_data_path,rate_data_col,'\t')

#print(rate_data.dtypes)

user_num=user_data.shape[0]
item_num=item_data.shape[0]

#把userid和itemid 按序排列 转为排序下标
def id2xid(id_list):
    id_list.sort()
    id2xid=dict(zip(id_list,range(0,len(id_list))))
    return id2xid

user_id2xid=id2xid(list(user_data['userID']))
item_id2xid=id2xid(list(item_data['itemID']))

#rate_data=rate_data.head(10000)

#Xu_train=rate_data['userID'].map(user_id2xid).values
#Xv_train=rate_data['itemID'].map(item_id2xid).values
#train_num=rate_data.shape[0]

#获取train test data
def get_train_test_data():
    train_num=train_data.shape[0]
    test_num=test_data.shape[0]

    Xu_train=train_data['userID'].map(user_id2xid).values
    Xv_train=train_data['itemID'].map(item_id2xid).values

    Xu_train=np.reshape(Xu_train,[train_num,1]) #一定到加上前边的维度 不能用None
    Xv_train=np.reshape(Xv_train,[train_num,1])
    y_train=np.reshape(train_data['rating'].values,[train_num,1])

    Xu_test=test_data['userID'].map(user_id2xid).values
    Xv_test=test_data['itemID'].map(item_id2xid).values

    Xu_test=np.reshape(Xu_test,[test_num,1]) #一定到加上前边的维度 不能用None
    Xv_test=np.reshape(Xv_test,[test_num,1])
    y_test=np.reshape(test_data['rating'].values,[test_num,1])
    return Xu_train,Xv_train,y_train,Xu_test,Xv_test,y_test

Xu_train,Xv_train,y_train,Xu_test,Xv_test,y_test=get_train_test_data()

from model_mf import Model
model=Model(user_num,item_num)
model.fit(Xu_train,Xv_train,y_train,Xu_test,Xv_test,y_test)

#tensorboard --logdir=E:/gjfCode/tf_pratice/tf_pratice/logs --host=127.0.0.1
