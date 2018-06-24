
# coding: utf-8

# In[2]:

# 필요한 모듈 불러오기
import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split

# In[3]:

# Working Space 지정
os.chdir("C:/Users/Jiseong Yang/Documents/양지성/Scholar/정규학기/3-2/데이터마이닝실습/프로젝트/777랜덤데이터셋")
os.getcwd()

# 데이터셋 불러오기 및 훈련/검정데이터 분리
xy = np.loadtxt('bank_dataset.csv',delimiter=',',dtype=np.float32)
xy_train,xy_test = train_test_split(xy,test_size=0.2)

#%% 훈련데이터 표준화 스케일링
xy_1 = xy_train[:,0:5]
xy_2 = xy_train[:,5:10]
xy_standadized_skl = StandardScaler().fit_transform(xy_1)
xy_train = np.concatenate((xy_standadized_skl,xy_2), axis=1)

#%% 검정데이터 표준화 스케일링
xy_3 = xy_test[:,0:5]
xy_4 = xy_test[:,5:10]
xy_standadized_sk2 = StandardScaler().fit_transform(xy_3)
xy_test = np.concatenate((xy_standadized_sk2,xy_4), axis=1)

#%%
x_data_train = xy_train[:,:-1]
y_data_train = xy_train[:,[-1]]
x_data_test = xy_test[:,:-1]
y_data_test = xy_test[:,[-1]]

# 훈련데이터와 검정데이터를 각각 csv 파일로 저장(주석처리)
# np.savetxt("bank4_scaled_train.csv", xy_train, delimiter=',')
# np.savetxt("bank4_scaled_test.csv", xy_test, delimiter=',')

# 정상적으로 저장되었는지 확인(주석처리)
#train = np.loadtxt("bank_train_777.csv", delimiter = ',', dtype = np.float32)
#test = np.loadtxt("bank_test_777.csv", delimiter = ',', dtype = np.float32)
#x_data_train = train[:,:-1]
#y_data_train = train[:,[-1]]
#x_data_test = train[:,:-1]
#y_data_test = train[:,[-1]]


# In[5]:

# 하이퍼파라미터(학습률, dropout, thresghold) 지정
tf.reset_default_graph()
learning_rate = 0.003
keep_prob = tf.placeholder_with_default(0.6, shape=())
threshold = tf.placeholder_with_default(0.65, shape=())

# In[6]:

# Placeholders 지정
X = tf.placeholder(tf.float32, [None, 9])
Y = tf.placeholder(tf.float32, [None, 1])


# In[7]:

# Input과 output 갯수 설정
num_in = 9
num_hidden = 45
num_out = 1
    
# Layer 1
W1 = tf.get_variable("weight1", shape=[num_in,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("bias1", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L1 = tf.nn.leaky_relu(tf.matmul(X, W1) + b1)
L1 =tf.nn.dropout(_L1, keep_prob)

# Layer 2
W2 = tf.get_variable("weight2", shape=[num_hidden,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("bias2", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L2 = tf.nn.leaky_relu(tf.matmul(L1, W2) + b2)
L2 =tf.nn.dropout(_L2, keep_prob)
 
# Layer 3
W3 = tf.get_variable("weight3", shape=[num_hidden,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("bias3", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L3 = tf.nn.leaky_relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(_L3, keep_prob)

# Layer 4
W4 = tf.get_variable("weight4", shape=[num_hidden,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("bias4", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L4 = tf.nn.leaky_relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(_L4, keep_prob)

# Layer 5
W5 = tf.get_variable("weight5", shape=[num_hidden,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable("bias5", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L5 = tf.nn.leaky_relu(tf.matmul(L4, W5) + b5)
L5 = tf.nn.dropout(_L5, keep_prob)

# Layer 6
W6 = tf.get_variable("weight6", shape=[num_hidden,num_out], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.get_variable("bias6", shape=[num_out], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
hypothesis = tf.sigmoid(tf.matmul(L5, W6) + b6)
hypothesis = tf.clip_by_value(hypothesis,1e-1, 1-(1e-1))


# In[8]:

# cost 함수 정의와 optimizer 지정
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis > threshold, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


# In[9]:

# 데이터셋 학습
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    print("학습을 시작합니다.")
    for step in range(1001):
        sess.run(train, feed_dict={X: x_data_train, Y: y_data_train})
        if step % 50 == 0:
            print("step ",step, "cost ", sess.run(cost, feed_dict={
                  X: x_data_train, Y: y_data_train}))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data_test, Y: y_data_test, keep_prob: 1.0})
    
    # 결과 메세지 출력
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
    if a >= 0.9:
        print("\n정확도 90% 이상! 성공입니다.")
    elif a >= 0.8:
        print("\n정확도 90% 미만! 개선이 필요합니다.")
    else:
        print("\n정확도 80% 미만! 실패입니다.")
 
