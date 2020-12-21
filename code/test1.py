import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf #使用tf的v1版本
tf.disable_v2_behavior() #使用tf的v1版本

##Xu_train=np.reshape([1,2,3,4,5],[5,1])
##print(Xu_train)
#A=[[1,2],[3,4],[5,6]]
#a=tf.Variable(A)
#bid=np.reshape([1,2,1,1],(4,1))
#b=tf.nn.embedding_lookup(a,bid) 

#sess=tf.Session()
#sess.run(tf.global_variables_initializer())

#a,b=sess.run([a,b])
#print(b)
#print(b.shape)

#l=[1,2,3,4,5]
#print(l[3:5])

#A=tf.Variable(tf.random_normal([5]),name='edage_type_score')
#A=tf.nn.softmax(A)
#weights=tf.split(A,5,0)


#B=tf.Variable(tf.random_normal([5,3,3]))
#C=tf.Variable(tf.random_normal([3,3]))

#result=weights[0]*C
#result1=weights*B

#A=tf.Variable([[1],[3],[5]])
#result=tf.subtract(A,1)
#result=tf.one_hot(result,5)

#sess=tf.Session()
#sess.run(tf.global_variables_initializer())

#xx=sess.run(result)
#print(xx)


## 测试np矩阵相乘
#a = np.array([[1, 2], [3, 4],[5,6]]) # None * 2
#b = np.array([1,2]) # 1 * 2
#result=np.multiply(a, b)

#result=np.matmul(result,np.reshape([1,1],(2,1)))
#print(result)

# 测试np one-hot
#a=np.array([[1],[2],[5],[3]])
#a=a-1
#result=(np.arange(5)==a[:,None]).astype(np.integer)
#result=np.reshape(result,(a.shape[0],5))
#print(result)
#print(a.shape)
#print(result.shape)

# 测试np 取每一行最大值的Index
#a = np.array([[1, 2], [3, 4],[5,6]]) # None * 2
#result=a.argmax(axis=1) # axis等于哪一维度 那一维度就没了
#print(result)
#print(result.shape)

# 测试tf 2维矩阵和3维矩阵 乘法结果
a=tf.Variable(tf.random_normal([8,10]))
b=tf.Variable(tf.random_normal([10,10,5]))
c=tf.matmul(a,b)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

result=sess.run(c)
print(result.shape)
