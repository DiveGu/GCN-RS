import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow
import tensorflow.compat.v1 as tf #使用tf的v1版本
tf.disable_v2_behavior() #使用tf的v1版本
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error

class Model():

    def __init__(self,user_num,item_num,Au,Du,Av,Dv,
                 embedding_size=8,
                 epoch=300, batch_size=256,show_batch=100,
                 learning_rate=0.01,loss_type='mse'):
        #原始嵌入维度 user嵌入表 item嵌入表
        self.user_num=user_num
        self.item_num=item_num
        self.Au=Au # 用户*物品
        self.Du=Du # 用户*用户
        self.Av=Av # 物品*用户
        self.Dv=Dv # 物品*物品
        self.embedding_size=embedding_size
        self.epoch=epoch
        self.batch_size=batch_size
        self.show_batch=show_batch
        self.learning_rate=learning_rate
        self.loss_type=loss_type
        self.train_result=[]
        self._init_graph()

    #初始化参数
    def _initialize_weights(self):
        weights = dict()

        # 用户和物品的初始embeddings
        weights['user_embeddings'] = tf.Variable(
            tf.random_normal([self.user_num, self.embedding_size], 0.0, 1),name="user_embeddings")  

        weights['item_embeddings'] = tf.Variable(
            tf.random_uniform([self.item_num, self.embedding_size], 0.0, 1), name="item_embeddings") 

        # 用户节点和物品节点的 拉普拉斯 嵌入
        weights['user_L']=tf.Variable(self.Au,name='user_L')
        weights['item_L']=tf.Variable(self.Au,name='item_L')

        # gcn层的w b
        weights['gcn_w_1']=tf.Variable(
            tf.random_normal([self.embedding_size,self.embedding_size],0.0,0.1),name="gcn_w_1")

        weights['gcn_b_1']=tf.Variable(tf.zeros([1,self.embedding_size])+0.01,name='gcn_b_1')

        return weights

    #构造模型
    def _init_graph(self):
        self.uid=tf.placeholder(tf.int32,shape=[None,1],name='uid')
        self.iid=tf.placeholder(tf.int32,shape=[None,1],name='iid')
        self.label = tf.placeholder(tf.int32, shape=[None, 1], name="label")  # None * 1
        self.weights=self._initialize_weights()

        # model
        #self.user_embeddings = tf.nn.embedding_lookup(self.weights["user_embeddings"],self.uid) # None * 1 * K
        #self.item_embeddings=tf.nn.embedding_lookup(self.weights["item_embeddings"],self.iid) # None * 1 * K

        # 1:h0嵌入表
        self.user_embeddings = self.weights['user_embeddings'] # U * K 所有不重复用户的嵌入
        self.item_embeddings=self.weights['item_embeddings'] # V * K 所有不重复物品的嵌入

        # 2:得到h1嵌入表
        # H_1=f(L*H_0*W_1)
        # Hu_1=f(Lu * Hv_0 *W_1)

        # [U * V] [V * K] [K * K]
        self.user_gcn1=tf.matmul(tf.cast(self.Au,tf.float32),self.item_embeddings)  # [U * V] [V * K]
        self.user_gcn1=tf.nn.relu(tf.matmul(self.user_gcn1,self.weights['gcn_w_1'])+self.weights['gcn_b_1'])  # [U * K] [K * K]

        # [V * U] [U * K] [K * K] 
        self.item_gcn1=tf.matmul(tf.cast(self.Av,tf.float32),self.user_embeddings)  # [V * U] [U * K] #类型float64改成float32
        self.item_gcn1=tf.nn.relu(tf.matmul(self.item_gcn1,self.weights['gcn_w_1'])+self.weights['gcn_b_1'])  # [V * K] [K * K]

        # 3：取h1嵌入 做预测
        self.user_h1=tf.nn.embedding_lookup(self.user_gcn1,self.uid) # None * 1 * K
        self.item_h1=tf.nn.embedding_lookup(self.item_gcn1,self.iid) # None * 1 * K

        # 3-1:直接相乘 预测
        self.out=tf.multiply(self.user_h1, self.item_h1) # None * 1 * K
        self.out=tf.reduce_sum(self.out,axis=2,keepdims=False) # None * 1

        # 3-2：拼接接个全连接
        #self.user_h1=tf.squeeze(self.user_h1,axis=1) # None * K
        #self.item_h1=tf.squeeze(self.item_h1,axis=1) # None * K
        #self.out=tf.concat([self.user_h1,self.item_h1],axis=1) # None * 2K
        #self.out=tf.layers.dense(self.out,1,tf.nn.relu)

        # loss
        if self.loss_type == "logloss":
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.label, self.out)
        elif self.loss_type == "mse":
            self.out=tf.clip_by_value(self.out,1,5)
            #self.loss = tf.nn.l2_loss(tf.subtract(tf.cast(self.label,tf.float32), tf.cast(self.out,tf.float32)))*2/2048 #真正的应该 *2/N
            self.loss = tf.losses.mean_squared_error(self.label, self.out) #mse
            #tf.summary.scalar('train_loss',tf.sqrt(tf.reduce_mean(self.loss)))#画loss

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

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

    #通过 X_valid,y_valid 获取验证指标
    def evaluate(self, Xu, Xv, y):
        y_pred = self.predict(Xu, Xv)
        return np.sqrt(mean_squared_error(y, y_pred))

    #训练模型
    def fit(self,Xu_train, Xv_train, y_train, Xu_valid=None,Xv_valid=None, y_valid=None,):
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xu_train, Xv_train, y_train) #同样的方式打乱这三个数据集
            total_batch = int(len(y_train) / self.batch_size)#不需要向上取整吗？
            for i in range(total_batch):
                #按照每个epoch里的batch次数 依次获得对应Index的训练数据 
                Xu_batch, Xv_batch, y_batch = self.get_batch(Xu_train, Xv_train, y_train, self.batch_size, i)                
                #train_mse,train_graph=self.fit_on_batch(Xu_batch, Xv_batch, y_batch)#将每次的batch进行训练
                train_mse=self.fit_on_batch(Xu_batch, Xv_batch, y_batch)#将每次的batch进行训练
                train_mse=train_mse**0.5
                if((i+1)%self.show_batch==0 or i==total_batch-1):
                    train_result = self.evaluate(Xu_train, Xv_train, y_train)
                    valid_result = self.evaluate(Xu_valid, Xv_valid, y_valid)
                    #self.writer.add_summary(train_graph,draw_num)
                    print("epoch{}/{},batch{}/{},train-result={:.4f},valid-result={:.4f} [{:.1f}s]" .format(epoch + 1,self.epoch,i+1 ,total_batch,train_mse,valid_result, time() - t1))

#C:\Users\Administrator\AppData\Roaming\Python\Python37\Scripts
#tensorboard --logdir=E:/gjfCode/tf_pratice/tf_pratice/logs --host=127.0.0.1


