import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf #使用tf的v1版本
tf.disable_v2_behavior() #使用tf的v1版本
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error,precision_score

class Model():

    def __init__(self,user_num,item_num,L_list,
                 embedding_size=10,rate_type=5,
                 epoch=300, batch_size=256,show_batch=1000,
                 learning_rate=0.01,loss_type='mse',reg_l2=0.01):
        #原始嵌入维度 user嵌入表 item嵌入表
        self.user_num=user_num
        self.item_num=item_num
        self.L_list=L_list #拉普拉斯算子 列表 每个评分一个L 一个W
        self.L_num=len(self.L_list)
        self.embedding_size=embedding_size
        self.rate_type=rate_type
        self.epoch=epoch
        self.batch_size=batch_size
        self.show_batch=show_batch
        self.learning_rate=learning_rate
        self.loss_type=loss_type
        self.reg_l2=reg_l2
        self.train_result=[]
        self._init_graph()

    #初始化参数
    def _initialize_weights(self):
        weights = dict()

        # 所有节点的初始embeddings
        weights['h0_embeddings'] = tf.Variable(
            tf.random_uniform([self.user_num+self.item_num, self.embedding_size], -1,1),name="h0_embeddings")  

        #weights['h0_embeddings'] = tf.Variable(
        #    tf.random_normal([self.user_num+self.item_num, self.embedding_size], 0,0.1),name="h0_embeddings")  

        # gcn层的w b 每一种评分对应一个 w （后续每一层 每一种对应一个 w 目前只有一层卷积）
        weights['gcn_w_1']=[] # 这个1表示 第一层卷积的w集合
        for i in range(0,self.L_num):
            weights['gcn_w_1'].append(tf.Variable(
                tf.random_normal([self.embedding_size,self.embedding_size],0,0.01),name='gcn_w_L_{}'.format(i)))

        weights['gcn_b_1']=tf.Variable(tf.zeros([1,self.embedding_size])+0.01,name='gcn_b_1') #这个1 表示是第一层卷积的b
        
        # 不同类型评分的邻居 聚合权重
        weights['edage_type_score']=tf.Variable(tf.random_normal([self.L_num]),name='edage_type_score')

        # 聚合h0,h1..的权重
        weights['stack_score']=tf.Variable(tf.random_normal([2,1]),name='stack_score')

        weights['fc_w']=tf.Variable(tf.random_normal([self.embedding_size*2,self.embedding_size]),name='fc_w')
        weights['fc_b']=tf.Variable(tf.zeros([1,self.embedding_size])+0.01,name='fc_b')

        if self.loss_type == "logloss":
            # 最后全连接层 预测
            weights['fc_class_w']=tf.Variable(tf.random_normal([self.embedding_size*2,self.rate_type]),name='fc_class_w')
            weights['fc_class_b']=tf.Variable(tf.zeros([1,self.rate_type])+0.01,name='fc_class_b')
            # 不用全连接层 用  [K * K * 5] 的参数
            weights['Q']=tf.Variable(tf.random_normal([self.embedding_size,self.embedding_size,self.rate_type]),name='fc_class_w')

        return weights

    #构造模型
    def _init_graph(self):
        self.uid=tf.placeholder(tf.int32,shape=[None,1],name='uid')
        self.iid=tf.placeholder(tf.int32,shape=[None,1],name='iid')
        if(self.loss_type=='mse'):
            self.label = tf.placeholder(tf.int32, shape=[None, 1], name="label")  # None * 1
        elif(self.loss_type=='logloss'):
            self.label = tf.placeholder(tf.int32, shape=[None, self.rate_type], name="label")  # None * 5
        self.weights=self._initialize_weights()

        # model
        #H_1=f(L*H_0*W_1)
        # 1:根据h0 得到h1嵌入表
        self.weights['edage_type_score']=tf.nn.softmax(self.weights['edage_type_score'])
        edage_weights=tf.split(self.weights['edage_type_score'],self.L_num,0)
        for i in range(self.L_num):
            temp=tf.matmul(tf.cast(self.L_list[i],tf.float32),self.weights['h0_embeddings']) # [U+V * U+V] [U+V * K]
            # L和每层w的个数是相同的 对应相乘
            temp=tf.matmul(temp,self.weights['gcn_w_1'][i])  # [U+V * K] [K * K] 
            # 不同评分的嵌入对用户 物品 贡献不同
            temp=edage_weights[i]*temp
            if(i==0):
                self.h1_embeddings=temp
            else:
                self.h1_embeddings=tf.add(self.h1_embeddings,temp)

        self.h1_embeddings=tf.nn.tanh(self.h1_embeddings+self.weights['gcn_b_1'])  # 激活函数(每种类型卷积之后相加 + b)

        # 2：取h0嵌入
        self.user_h0=tf.nn.embedding_lookup(self.weights['h0_embeddings'],self.uid) # None * 1 * K
        self.item_h0=tf.nn.embedding_lookup(self.weights['h0_embeddings'],self.iid) # None * 1 * K

        # 2：取h1嵌入 做预测
        self.user_h1=tf.nn.embedding_lookup(self.h1_embeddings,self.uid) # None * 1 * K
        self.item_h1=tf.nn.embedding_lookup(self.h1_embeddings,self.iid) # None * 1 * K

        # 3: h0和h1嵌入做聚合
        # 3-1：拼接聚合
        #self.user=tf.concat([self.user_h0,self.user_h1],axis=2) # None * 1 * 2K
        #self.item=tf.concat([self.item_h0,self.item_h1],axis=2) # None * 1 * 2K

        # 分别接一个fc fc的参数相同
        #self.user=tf.nn.tanh(tf.matmul(self.user,self.weights['fc_w'],)+self.weights['fc_b'])
        #self.item=tf.nn.tanh(tf.matmul(self.item,self.weights['fc_w'],)+self.weights['fc_b'])
        #self.item=tf.layers.dense(self.user,self.embedding_size,tf.nn.tanh)

        # 3-2：sum聚合
        #self.user=tf.add(self.user_h0,self.user_h1) # None * 1 * K
        #self.item=tf.add(self.item_h0,self.item_h1) # None * 1 * K

        # 3-3：加权聚合
        self.weights['stack_score']=tf.nn.softmax(self.weights['stack_score']) # 2 * 1
        self.user=tf.concat([self.user_h0,self.user_h1],axis=1) # None * 2 * K
        self.user=tf.transpose(self.user,[0,2,1]) # None * K * 2
        self.item=tf.concat([self.item_h0,self.item_h1],axis=1) # None * 2 * K
        self.item=tf.transpose(self.item,[0,2,1]) # None * K * 2
        self.user=tf.matmul(self.user,self.weights['stack_score']) # [None * K * 2] [2 * 1]
        self.item=tf.matmul(self.item,self.weights['stack_score']) # [None * K * 2] [2 * 1]
        self.user=tf.transpose(self.user,[0,2,1]) # None * 1 * K
        self.item=tf.transpose(self.item,[0,2,1]) # None * 1 * K

        # 只使用h1嵌入
        #self.user, self.item=self.user_h1, self.item_h1

        # 使用h0和h1

        # loss
        if self.loss_type == "logloss":
            # 1：拼接user-item 最后接一个全连接层
            #self.out=tf.concat([self.user,self.item],axis=2) # None * 1 * 2K
            #self.out=tf.squeeze(self.out,axis=1) # None * 2K
            #self.out=tf.nn.softmax(tf.matmul(self.out,self.weights['fc_class_w'])+self.weights['fc_class_b']) # None * 5
            # 2：user W item
            self.user=tf.squeeze(self.user,axis=1) # None * K
            self.item=tf.squeeze(self.item,axis=1) # None * K
            self.user_out=tf.matmul(self.user,self.weights['Q']) # [None * K] [K * K * 5] 得 [K * None * 5]
            self.item_out=tf.matmul(self.item,tf.transpose(self.weights['Q'],[1,0,2])) # [None * K] [K2 * K1 * 5] 得 [K * None * 5]

            self.out=tf.multiply(self.user_out,self.item_out) # [None * K * 5] 
            self.out=tf.nn.softmax(tf.reduce_sum(self.out,axis=0,keepdims=False)) # None * 5

            #self.loss = tf.losses.log_loss(self.label, self.out) # 交叉熵
            self.loss=tf.nn.softmax_cross_entropy_with_logits_v2(self.label,self.out) # 交叉熵 注：此函数只适用于多分类中的单分类

        elif self.loss_type == "mse":
            self.out=tf.multiply(self.user, self.item) # None * 1 * K
            self.out=tf.reduce_sum(self.out,axis=2,keepdims=False) # None * 1
            self.out=tf.clip_by_value(self.out,1,5)
            self.loss = tf.nn.l2_loss(tf.subtract(tf.cast(self.label,tf.float32), tf.cast(self.out,tf.float32))) 
            #self.loss = tf.losses.mean_squared_error(self.label, self.out) #mse
            #tf.summary.scalar('train_loss',tf.sqrt(tf.reduce_mean(self.loss)))#画loss

        self.loss=self.loss+self.reg_l2*tf.nn.l2_loss(self.weights['h0_embeddings'])+self.reg_l2*tf.nn.l2_loss(self.h1_embeddings)

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
    def fit_on_batch(self, Xu,Xv, y):
        feed_dict = {self.uid: Xu,
                     self.iid: Xv,
                     self.label: y,}
        #loss, train, merged = self.sess.run((self.loss, self.train, self.merged), feed_dict=feed_dict)
        loss, train= self.sess.run((self.loss, self.train), feed_dict=feed_dict)
        return loss

    #传入X 预测y
    def predict(self, Xu, Xv):
        # dummy y
        if(self.loss_type=='mse'):
            dummy_y = [1] * len(Xu)
            dummy_y=np.reshape(dummy_y,[len(Xu),1])
        elif(self.loss_type=='logloss'):
            dummy_y = [1] * self.rate_type * len(Xu)
            dummy_y=np.reshape(dummy_y,[len(Xu),self.rate_type])
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
                if(self.loss_type=='mse'):
                    y_pred = np.reshape(batch_out, (num_batch,))
                elif(self.loss_type=='logloss'):
                    y_pred = np.reshape(batch_out, (num_batch,self.rate_type))
            else:
                if(self.loss_type=='mse'):
                    y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))#把y_pred拼接起来
                elif(self.loss_type=='logloss'):
                    y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,self.rate_type))))#把y_pred拼接起来

            batch_index += 1
            Xu_batch, Xv_batch, y_batch = self.get_batch(Xu, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred

    #通过 X_valid,y_valid 获取验证指标
    def evaluate_rmse(self, Xu, Xv, y):
        y_pred = self.predict(Xu, Xv)
        # 如果是logloss 
        if(self.loss_type=='logloss'):
            # 1：转化 y_pred 从 None * 5 转化成 None * 1
            # 1-1：取概率最大的rate作为predict
            #y_pred=np.argmax(y_pred,axis=1) # None * 5 取第2维度最大值的index 变成 None维度
            #y_pred=np.reshape(y_pred,(y.shape[0],1)) # None * 1 升一维度
            #y_pred=y_pred+1
            # 1-2：每种概率乘rate 求期望
            ratings=np.array([i for i in range(1,self.rate_type+1)]) # [1,2,3,4,5]
            y_pred=np.multiply(y_pred,ratings) # 每种得分*对应概率
            y_pred=np.matmul(y_pred,np.reshape([1]*self.rate_type,(self.rate_type,1)))# 求和 从 None * 5 变成 None * 1
           
            # 2：y从one-hot转化回评分 从 None * 5 转化成 None * 1
            y=np.argmax(y,axis=1) # None * 5 取第2维度最大值的index 变成 None维度
            y=np.reshape(y,(y.shape[0],1)) # None * 1 升一维度
            y=y+1 # index+1是真正的评分

        return np.sqrt(mean_squared_error(y, y_pred))


    #通过 X_valid,y_valid 获取验证指标 按照分类准确率来算
    def evaluate(self, Xu, Xv, y):
        y_pred = self.predict(Xu, Xv)
        if(self.loss_type=='mse'):
            return np.sqrt(mean_squared_error(y, y_pred))
        # 如果是logloss 
        if(self.loss_type=='logloss'):
            # 1：转化 y_pred 从 None * 5 转化成 None * 1
            # 1-1：取概率最大的rate作为predict
            y_pred=np.argmax(y_pred,axis=1) # None * 5 取第2维度最大值的index 变成 None维度
            y_pred=np.reshape(y_pred,(y.shape[0],1)) # None * 1 升一维度
            y_pred=y_pred+1
            # 1-2：每种概率乘rate 求期望

            # 2：y从one-hot转化回评分 从 None * 5 转化成 None * 1
            y=np.argmax(y,axis=1) # None * 5 取第2维度最大值的index 变成 None维度
            y=np.reshape(y,(y.shape[0],1)) # None * 1 升一维度
            y=y+1 # index+1是真正的评分

            return precision_score(y,y_pred,average='weighted')
    
    #numpy 转化成one-hot
    def np_one_hot(self,x):
        a=x-1
        result=(np.arange(self.rate_type)==a[:,None]).astype(np.integer) # None * 1 * 5
        result=np.reshape(result,(a.shape[0],5)) # None * 5
        return result

    #训练模型
    def fit(self,Xu_train, Xv_train, y_train, Xu_valid=None,Xv_valid=None, y_valid=None,):
        # None * 1 转化成 None * 5
        if(self.loss_type=='logloss'):
            # 用tf.one_hot 两个变量都变成tf类型 了
            #y_train=tf.one_hot(tf.subtract(y_train,1),self.rate_type) # 转化成one-hot 先-1 再one-hot
            #y_valid=tf.one_hot(tf.subtract(y_valid,1),self.rate_type) # 转化成one-hot 先-1 再one-hot
            # 用np转化成one-hot
            y_train=self.np_one_hot(y_train)
            y_valid=self.np_one_hot(y_valid)

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xu_train, Xv_train, y_train) #同样的方式打乱这三个数据集
            total_batch = int(len(y_train) / self.batch_size)#不需要向上取整吗？
            for i in range(total_batch):
                #按照每个epoch里的batch次数 依次获得对应Index的训练数据 
                Xu_batch, Xv_batch, y_batch = self.get_batch(Xu_train, Xv_train, y_train, self.batch_size, i)                
                #train_mse,train_graph=self.fit_on_batch(Xu_batch, Xv_batch, y_batch)#将每次的batch进行训练
                loss=self.fit_on_batch(Xu_batch, Xv_batch, y_batch)#将每次的batch进行训练
                if((i+1)%self.show_batch==0 or i==total_batch-1):
                    #train_result = self.evaluate(Xu_train, Xv_train, y_train)
                    #valid_result = self.evaluate(Xu_valid, Xv_valid, y_valid)
                    train_result = self.evaluate_rmse(Xu_train, Xv_train, y_train)
                    valid_result = self.evaluate_rmse(Xu_valid, Xv_valid, y_valid)
                    #self.writer.add_summary(train_graph,draw_num)
                    print("epoch{}/{},batch{}/{},train-result={:.4f},valid-result={:.4f} [{:.1f}s]" .format(epoch + 1,self.epoch,i+1 ,total_batch,train_result,valid_result, time() - t1))

#C:\Users\Administrator\AppData\Roaming\Python\Python37\Scripts
#tensorboard --logdir=E:/gjfCode/tf_pratice/tf_pratice/logs --host=127.0.0.1


