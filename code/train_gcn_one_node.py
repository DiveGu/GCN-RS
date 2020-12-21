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

user_id2xid=id2xid(list(user_data['userID'])) #uid:下标
item_id2xid=id2xid(list(item_data['itemID'])) #iid:下标

#所有节点看作一类 构造uid和iid转化下标的字典
uid_id2xid=user_id2xid #user对应下标：0,user_num-1
iid_id2xid=dict(zip(list(item_data['itemID']),range(user_num,user_num+item_num))) #item对应下标：user_num,user_num+item_num-1

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
    node_num=len(uid_list)+len(iid_list)
    # 1所有用户节点和所有物品节点构造一个矩阵
    # A矩阵中索引对应用户或物品的下标
    A=np.zeros((node_num,node_num))
    # 2从train的交互数据中构造邻接矩阵A
    train_data['userID']=train_data['userID'].map(uid_id2xid)
    train_data['itemID']=train_data['itemID'].map(iid_id2xid)
    for index,item in train_data.iterrows():
        A[item['userID']][item['itemID']]+=1
    # 3构造度矩阵D
    D=np.zeros((node_num,node_num))
    # 计算uid的出现次数
    cnt=train_data['userID'].value_counts()
    for k in cnt.index:
        D[k][k]=cnt[k]
    # 计算iid的出现次数
    cnt=train_data['itemID'].value_counts()
    for k in cnt.index:
        D[k][k]=cnt[k]
    D_normal=np.zeros((node_num,node_num))
    for i in range(0,node_num):
        for j in range(0,node_num):
            if(D[i][j]>0):
                D_normal[i][j]=1/(D[i][j]**0.5)
            else:
                D_normal[i][j]=0

    L=np.dot(D_normal,A)
    L=np.dot(L,D_normal)
    L=np.identity(node_num)+L
    return A,D,L

# 获取不同评分的L之和
def get_different_L(uid_list,iid_list,train_data):
    L=[]
    for rating in range(1,6):
        L_single=get_graph_martix(uid_list,iid_list,train_data[train_data['rating']==rating])[2]
        L.append(L_single)
    return L

#把获得的矩阵保存
def save_graph_martix(save_path):
    L=get_different_L(list(user_data['userID']),list(item_data['itemID']),train_data)
    # save
    with open(save_path, 'wb') as f:
        pickle.dump(L, f)

#加载图矩阵
def load_graph_martix(save_path):
    with open(save_path, 'rb') as f:
        L=pickle.load(f)
    return L

#from model_gcn_one_node import Model
#save_path='./data_save/ml_martix_L_list.data'
##save_graph_martix(save_path)
#L_list=load_graph_martix(save_path)
#model=Model(user_num,item_num,L_list)
#model.fit(Xu_train,Xv_train,y_train,Xu_test,Xv_test,y_test)

from sklearn.decomposition import PCA,KernelPCA
from sklearn import manifold
import matplotlib.pyplot as plt

# 从uid_list挑选出物品集合
def select_u_i(uid_list):
    udict={}
    iid_list=[]
    for uid in uid_list:
        iid=train_data[train_data['userID']==uid]['itemID']
        iid=[x+user_num for x in iid ]
        udict[uid]=iid
        iid_list.extend(iid)
    iid_list=set(iid_list)

    return udict

#user_group=train_data.groupby('userID')['itemID'].count()
#candicate_user=[] #选出看电影数较少的用户 可视化好选
#for k,v in user_group.items():
#    if(v>10 and v<20):
#        candicate_user.append(k)
#print(candicate_user)
#np.random.shuffle(candicate_user)
#uid_list=candicate_user[:5]
#print(uid_list)

def pca_2d():
    uid_list=[306, 431, 78, 202, 120]
    uid_list=[306, 32,78]
    colors = ['navy', 'green', 'darkorange','turquoise','grey'] # 每个uid一个颜色
    udict=select_u_i(uid_list)
    embeddings=pd.read_csv('./ml-100k-result/embedding_final_mf.csv',index_col=0)
    #embeddings=pd.read_csv('./ml-100k-result/embedding_final_gcn_layer=1.csv',index_col=0) #一定记得index_col=0 忽略掉index列
    X=embeddings.values
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_2d = tsne.fit_transform(X)
    #df=pd.DataFrame(X_2d)
    #df.to_csv('./X_2d_tsne.csv')
    #pca = PCA(n_components=2)
    #X_2d = pca.fit(X).transform(X)

    plt.figure()
    for i in range(0,len(uid_list)):
        plt.scatter(X_2d[uid_list[i]][0],X_2d[uid_list[i]][1],color=colors[i],marker='*')#用户点
        for iid in udict[uid_list[i]]:
            plt.scatter(X_2d[iid][0],X_2d[iid][1],color=colors[i],s=20) #物品点
            plt.plot([X_2d[uid_list[i]][0],X_2d[iid][0]],[X_2d[uid_list[i]][1],X_2d[iid][1]],linewidth=0.1,color=colors[i])# 连线
    plt.legend(uid_list,labels=uid_list)
    plt.show()

pca_2d()
