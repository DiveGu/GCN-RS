import numpy as np
import pandas as pd
import pickle

def get_init_data(path,col,sep):
    data=pd.read_csv(path,sep = sep,encoding='latin-1',header=None)
    data.columns=col
    return data

user_data_path='F:/data/ml-100k/u.user'
item_data_path='F:/data/ml-100k/u.item'
rate_data_path='F:/data/ml-100k/u.data'
train_data_path='F:/data/ml-100k/u1.base'
test_data_path='F:/data/ml-100k/u1.test'

user_data_col=['userID','age','gender','occupation','zipCode']

item_data_col=['itemID', 'movieTitle', 'releaseDate', 'videoReleaseDate',
              'IMDB_URL' ,'unknown', 'Action', 'Adventure', 'Animation',
              'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western']

rate_data_col=['userID', 'itemID', 'rating', 'timestamp']


#读取原始的 用户数据 物品数据 交互数据
user_data=get_init_data(user_data_path,user_data_col,'|')
item_data=get_init_data(item_data_path,item_data_col,'|')

#读取 train 和 test 数据
train_data=get_init_data(train_data_path,rate_data_col,'\t')
test_data=get_init_data(test_data_path,rate_data_col,'\t')


user_num=user_data.shape[0]
item_num=item_data.shape[0]

#把userid和itemid 按序排列 转为排序下标
def id2xid(id_list):
    id_list.sort()
    id2xid=dict(zip(id_list,range(0,len(id_list))))
    return id2xid

user_id2xid=id2xid(list(user_data['userID']))
item_id2xid=id2xid(list(item_data['itemID']))

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

#获取用户和物品节点的邻接矩阵 度矩阵 
def get_graph_martix(uid_list,iid_list,train_data):
    # 1所有用户节点和所有物品节点构造一个 or 用户和物品构造1个 转置作为另一个
    A=pd.DataFrame(np.zeros((len(uid_list),len(iid_list))),index=uid_list,columns=iid_list)
    # 2从train的交互数据中构造邻接矩阵A
    for index,item in train_data.iterrows():
        A.loc[item['userID']][item['itemID']]+=1
    # 3构造度矩阵D
    D=pd.DataFrame(np.zeros((len(uid_list),len(uid_list))),index=uid_list,columns=uid_list)
    # 计算uid的出现次数
    cnt=train_data['userID'].value_counts()
    for k in cnt.index:
        D.loc[k][k]=cnt[k]
    return A.values,D.values

#把获得的矩阵保存
def save_graph_martix(save_path):
    Au,Du=get_graph_martix(list(user_data['userID']),list(item_data['itemID']),train_data)
    temp=train_data
    temp.columns=['itemID', 'userID', 'rating', 'timestamp']# iid和uid列名调换 
    Av,Dv=get_graph_martix(list(item_data['itemID']),list(user_data['userID']),temp)
    # save
    with open(save_path, 'wb') as f:
        pickle.dump([Au,Du,Av,Dv], f)

#加载图矩阵
def load_graph_martix(save_path):
    with open(save_path, 'rb') as f:
        Au,Du,Av,Dv=pickle.load(f)
    return Au,Du,Av,Dv


#from gcn1_model_gcn import Model
from model_gcn_two_nodes import Model
save_path='./data_save/ml_martix.data'
#save_graph_martix(save_path)
Au,Du,Av,Dv=load_graph_martix(save_path)
model=Model(user_num,item_num,Au,Du,Av,Dv)
model.fit(Xu_train,Xv_train,y_train,Xu_test,Xv_test,y_test)




