import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf #使用tf的v1版本
tf.disable_v2_behavior() #使用tf的v1版本
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import mean_squared_error

class Model():

    def __init__(self,user_num,item_num,L_list,
                 embedding_size=10,gcn_layer=1,gcn_activation='relu',
                 epoch=30, batch_size=256,show_batch=1000,final_mlp=True,
                 learning_rate=0.01,loss_type='mse',reg_l2=0.01):
        #原始嵌入维度 user嵌入表 item嵌入表
        self.user_num=user_num
        self.item_num=item_num
        self.L_list=L_list #拉普拉斯算子 列表 每个评分一个L 一个W
        self.L_num=len(self.L_list)
        self.embedding_size=embedding_size
        self.gcn_layer=gcn_layer
        self.gcn_activation=gcn_activation
        self.epoch=epoch
        self.batch_size=batch_size
        self.show_batch=show_batch
        self.final_mlp=final_mlp
        #self.user_item_mlp_same=user_item_mlp_same
        self.learning_rate=learning_rate
        self.loss_type=loss_type
        self.reg_l2=reg_l2
        self.train_result=[]
        self.valid_result=[]
        self._init_graph()

    #初始化参数
    def _initialize_weights(self):
        weights = dict()

        # 所有节点的初始embeddings
        weights['h0_embeddings'] = tf.Variable(
            tf.random_uniform([self.user_num+self.item_num, self.embedding_size], -1,1),name="h0_embeddings")  

        # 初始化每一层GCN的参数
        for layer in range(0,self.gcn_layer):
            weights['gcn_w_{}'.format(layer+1)]=[] # 1表示 第一层卷积的w集合
            # 每一层卷积的w有 [边的类型] 个
            for i in range(0,self.L_num):
                weights['gcn_w_{}'.format(layer+1)].append(tf.Variable(
                tf.random_normal([self.embedding_size,self.embedding_size],0,0.01),name='gcn_w_{}_L_{}'.format(layer+1,i+1)))

            weights['gcn_b_{}'.format(layer+1)]=tf.Variable(tf.zeros([1,self.embedding_size])+0.01,name='gcn_b_{}'.format(layer+1)) #1 表示是第一层卷积的b

        # 不同类型评分的邻居 聚合权重
        weights['edage_type_score']=tf.Variable(tf.random_normal([self.L_num]),name='edage_type_score')

        # 聚合h0,h1..的权重
        weights['stack_score']=tf.Variable(tf.random_normal([self.gcn_layer+1]),name='stack_score')

        weights['fc_w']=tf.Variable(tf.random_normal([self.embedding_size*2,self.embedding_size]),name='fc_w')
        weights['fc_b']=tf.Variable(tf.zeros([1,self.embedding_size])+0.01,name='fc_b')

        return weights

    #构造模型
    def _init_graph(self):
        self.uid=tf.placeholder(tf.int32,shape=[None,1],name='uid')
        self.iid=tf.placeholder(tf.int32,shape=[None,1],name='iid')
        self.label = tf.placeholder(tf.int32, shape=[None, 1], name="label")  # None * 1
        self.weights=self._initialize_weights()
        self.embedding_list=[self.weights['h0_embeddings']] # 存放每次更新后的嵌入表

        # model
        #
        # softmax 每种边的权重
        self.weights['edage_type_score']=tf.nn.softmax(self.weights['edage_type_score'])
        edage_weights=tf.split(self.weights['edage_type_score'],self.L_num,0)
        # 1 多层GCN
        for layer in range(0,self.gcn_layer):
            H_pre=self.embedding_list[-1] #获取上一层的嵌入表

            gcn_w=self.weights['gcn_w_{}'.format(layer+1)]
            gcn_b=self.weights['gcn_b_{}'.format(layer+1)]

            for i in range(self.L_num):
                temp=tf.matmul(tf.cast(self.L_list[i],tf.float32),H_pre) # [U+V * U+V] [U+V * K]
                # L和每层w的个数是相同的 对应相乘
                temp=tf.matmul(temp,gcn_w[i])  # [U+V * K] [K * K] 
                # 不同评分的嵌入对用户 物品 贡献不同
                #temp=edage_weights[i]*temp
                if(i==0):
                    H_cur=temp
                else:
                    H_cur=tf.add(H_cur,temp)
            # 这一层GCN结束
            # H_cur这一层GCN之后得到的嵌入表
            if(self.gcn_activation=='tanh'):
                H_cur=tf.nn.tanh(H_cur+gcn_b)
            elif(self.gcn_activation=='relu'):
                H_cur=tf.nn.relu(H_cur+gcn_b)
            elif(self.gcn_activation=='sigmoid'):
                H_cur=tf.nn.sigmoid(H_cur+gcn_b)

            #H_cur=tf.layers.dropout(H_cur, rate=0.5,training=True) 
            self.embedding_list.append(H_cur)

        # 2 计算最终嵌入层(拼接所有嵌入表 + MLP) 
        self.embedding_final=self.embedding_list[0]
        for i in range(1,len(self.embedding_list)):
            # 把所有的嵌入表拼接起来
            self.embedding_final=tf.concat([self.embedding_final,self.embedding_list[i]],axis=-1)
            
            
        ## 注意力机制得最终嵌入
        #self.weights['stack_score']=tf.nn.softmax(self.weights['stack_score'])
        #h_weights=tf.split(self.weights['stack_score'],self.gcn_layer+1,0)
        #for i in range(0,len(self.embedding_list)):
        #    if(i==0):
        #        self.embedding_final=h_weights[i]*self.embedding_list[i]
        #    else:
        #        self.embedding_final=self.embedding_final+h_weights[i]*self.embedding_list[i]

        # user和item mlp参数相同 不加激活函数
        #self.embedding_final=tf.layers.dense(self.embedding_final,self.embedding_size)

        # 3 取user item最终嵌入 做预测
        self.user=tf.nn.embedding_lookup(self.embedding_final,self.uid) # None * 1 * nK
        self.item=tf.nn.embedding_lookup(self.embedding_final,self.iid) # None * 1 * nK

        # concat之后是否需要加mlp
        if(self.final_mlp):
            self.user=tf.layers.dense(self.user,self.embedding_size)
            self.item=tf.layers.dense(self.item,self.embedding_size)

        self.out=tf.multiply(self.user, self.item) # None * 1 * K
        self.out=tf.reduce_sum(self.out,axis=2,keepdims=False) # None * 1

        # loss
        if self.loss_type == "logloss":
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.label, self.out)
        elif self.loss_type == "mse":
            self.out=tf.clip_by_value(self.out,1,5)
            self.loss = tf.nn.l2_loss(tf.subtract(tf.cast(self.label,tf.float32), tf.cast(self.out,tf.float32))) 
            #self.loss = tf.losses.mean_squared_error(self.label, self.out) #mse
            #tf.summary.scalar('train_loss',tf.sqrt(tf.reduce_mean(self.loss)))#画loss

        # loss+正则化
        for H in self.embedding_list:
            self.loss=self.loss+self.reg_l2*tf.nn.l2_loss(H)

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        #self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #self.merged=tf.summary.merge_all()
        #self.writer=tf.summary.FileWriter('./logs/',self.sess.graph)

    #获取一个batch的数据
    def get_batch(self, Xu, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xu[start:end], Xv[start:end], y[start:end]

    # 打乱训练数据
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    #给类传入真实的训练数据
    def fit_on_batch(self, Xu, Xv, y):
        feed_dict = {self.uid: Xu,
                     self.iid: Xv,
                     self.label: y,}
        #loss, train, merged = self.sess.run((self.loss, self.train, self.merged), feed_dict=feed_dict)
        loss, train,w1,w2= self.sess.run((self.loss, self.train,self.weights['stack_score'],self.weights['edage_type_score']), feed_dict=feed_dict)
        return loss,w1,w2

    #传入X 预测y
    def predict(self, Xu, Xv):
        # dummy y
        dummy_y = [1] * len(Xu)
        dummy_y=np.reshape(dummy_y,[len(Xu),1])
        batch_index = 0
        Xu_batch, Xv_batch, y_batch = self.get_batch(Xu, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xu_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.uid: Xu_batch,
                         self.iid: Xv_batch,
                         self.label: y_batch,}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))#把y_pred拼接起来

            batch_index += 1
            Xu_batch, Xv_batch, y_batch = self.get_batch(Xu, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred

    # 获取嵌入表 分析嵌入
    def save_embedding(self):
        uid_list=range(0,self.user_num)
        iid_list=range(self.user_num,self.user_num+self.item_num)
        # 1-用户最终表示
        #只要不run loss 不会对原model参数发生变化
        feed_dict = {self.uid: np.reshape(uid_list,(self.user_num,1)),#只需要用到的数据真实 即可
                     self.iid: np.reshape([1]*2,(2,1)),#用不到的数据 随便赋值 不影响 Num不一样也不影响
                     self.label: np.reshape([1]*2,(2,1)),}
        user_final=self.sess.run(self.user,feed_dict) # user_num,1,K
        user_final=np.squeeze(user_final,axis=1) #降除中间那个维度

        # 2-物品最终表示
        feed_dict = {self.uid: np.reshape([1]*2,(2,1)),#用不到的数据 随便赋值 不影响 Num不一样也不影响
                     self.iid: np.reshape(iid_list,(self.item_num,1)),#只需要用到的数据真实 即可
                     self.label: np.reshape([1]*2,(2,1)),}
        item_final=self.sess.run(self.item,feed_dict) # item_num,1,K
        item_final=np.squeeze(item_final,axis=1)
        # 3-拼接成嵌入表一样的形式 df保存
        df=np.concatenate((user_final,item_final),axis=0)
        df=pd.DataFrame(df)
        df.to_csv('./embedding_final_{}.csv'.format('gcn_layer={}'.format(self.gcn_layer)))

    #通过 X_valid,y_valid 获取验证指标
    def evaluate(self, Xu, Xv, y):
        y_pred = self.predict(Xu, Xv)
        return np.sqrt(mean_squared_error(y, y_pred))

    #训练模型
    def fit(self,Xu_train, Xv_train, y_train, Xu_valid=None,Xv_valid=None, y_valid=None,):
        best_valid_result=100
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xu_train, Xv_train, y_train) #同样的方式打乱这三个数据集
            total_batch = int(len(y_train) / self.batch_size)#不需要向上取整吗？
            for i in range(total_batch):
                #按照每个epoch里的batch次数 依次获得对应Index的训练数据 
                Xu_batch, Xv_batch, y_batch = self.get_batch(Xu_train, Xv_train, y_train, self.batch_size, i)                
                #self.fit_on_batch(Xu_batch, Xv_batch, y_batch) # 不需要获取return值
                _,w1,w2=self.fit_on_batch(Xu_batch, Xv_batch, y_batch)#将每次的batch进行训练
                if((i+1)%self.show_batch==0 or i==total_batch-1):
                    train_result = self.evaluate(Xu_train, Xv_train, y_train)
                    valid_result = self.evaluate(Xu_valid, Xv_valid, y_valid)
                    self.train_result.append(train_result)
                    self.valid_result.append(valid_result)
                    # 这个epoch之后表现更好的话 就保存嵌入 （后续分析嵌入用）
                    if(valid_result<best_valid_result):
                        best_valid_result=valid_result
                        self.save_embedding()

                    #self.save_result('./gcn_layer={}_loss={}.csv'.format(self.gcn_layer,self.loss_type))
                    #self.writer.add_summary(train_graph,draw_num)
                    print("epoch{}/{},batch{}/{},train-result={:.4f},valid-result={:.4f} [{:.1f}s]" .format(epoch + 1,self.epoch,i+1 ,total_batch,train_result,valid_result, time() - t1))
                    #print(w1)
                    #print(w2)


    # 保存每次epoch结束的评价结果
    def save_result(self,path):
        df=pd.DataFrame()
        df['train-result']=self.train_result
        df['valid-result']=self.valid_result
        df.to_csv(path)

#C:\Users\Administrator\AppData\Roaming\Python\Python37\Scripts
#tensorboard --logdir=E:/gjfCode/tf_pratice/tf_pratice/logs --host=127.0.0.1


