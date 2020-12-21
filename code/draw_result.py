import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set(style='darkgrid')
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
matplotlib.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus']=False


# 分析嵌入
# 1 根据df挑5个用户 找到交互过的物品
# 2 pca把所有嵌入变成两维
# 3 画图

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

def pca_2d():
    uid_list=[1,2,3,4,5]
    colors = ['navy', 'green', 'darkorange','turquoise','grey'] # 每个uid一个颜色
    udict=select_u_i(uid_list)

    embeddings=pd.read_csv('./embedding_final_mf.csv')
    #embeddings=pd.read_csv('./embedding_final_gcn_layer=1.csv')
    X=embeddings.values
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_2d = tsne.fit_transform(X)

    #pca = PCA(n_components=2)
    #X_2d = pca.fit(X).transform(X)

    plt.figure()
    for i in range(0,len(uid_list)):
        plt.scatter(X_2d[uid_list[i]][0],X_2d[uid_list[i]][1],color=colors[i],marker='*')
        for iid in udict[uid_list[i]]:
            plt.scatter(X_2d[iid][0],X_2d[iid][1],color=colors[i])
    plt.show()

#pca_2d()


def draw_loss(data_path):
    #画loss曲线
    mdata=pd.read_excel(data_path)

    #cols=['mf','gcn_l=0','gcn_l=1','gcn_l=2']
    cols=['mf','gcn_l=1','gcn_l=2']

    ax = sns.lineplot( mdata['ind'],mdata['mf'],linewidth=1.5)
    #ax = sns.lineplot( mdata['ind'],mdata['gcn_l=0'],linewidth=1.5)
    ax = sns.lineplot( mdata['ind'],mdata['gcn_l=1'],linewidth=1.5)
    ax = sns.lineplot( mdata['ind'],mdata['gcn_l=2'],linewidth=1.5)

    ax.legend(labels=cols,fontsize=14)
    #plt.ylim((0.8, 1.5))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('epoch',fontdict={'color': 'black',
                                 'weight': 'normal',
                                 'size': 18})
    plt.ylabel('RMSE',fontdict={'color': 'black',
                                 'weight': 'normal',
                                 'size': 18})

    #ax = sns.lineplot( mdata['ind'],mdata['mf'])
    #ax = sns.lineplot( mdata['ind'],mdata['gcn_l=1'])

    #ax.legend(labels=["mf","gcn_l=1"])

    ##plt.ylim((0.92, 1.1))

    #plt.xlabel('epoch')
    #plt.ylabel('rmse')

    plt.show()

data_path="./ml-100k-result/valid.xlsx"
draw_loss(data_path)
