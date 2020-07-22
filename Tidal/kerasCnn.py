"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 6 - CNN example

# to try tensorflow, un-comment following two lines
# import os
# os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


import tensorflow as tf
from keras import backend as K


# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP / P


# 接着在模型的compile中设置metrics
# 如下例子，我用的是RNN做分类


# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
#df = pd.read_csv('E:\\G-1149\\trafficCongestion\\训练数据\\4小时文件\\trainData\\Sample15_2.csv', header=0)
df = pd.read_csv('E:\\G-1149\\trafficCongestion\\训练数据\\trainData\\Sample15_1.csv', header=0)
data_sample = np.array(df).astype(float)
df = pd.read_csv('E:\\G-1149\\trafficCongestion\\训练数据\\trainData\\label.csv', header=0)
#df = pd.read_csv('E:\\G-1149\\trafficCongestion\\训练数据\\4小时文件\\trainData\\label.csv', header=0)
data_label = np.array(df).astype(float)
X_train, X_test, y_train, y_test = train_test_split(data_sample,data_label,test_size=0.3, random_state=0)

# data pre-processing
#-1是样本的个数，1是chanel,代表高度
X_train = X_train.reshape(-1, 1,32, 2)/3.
X_test = X_test.reshape(-1, 1,32, 2)/3.
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 32, 2),
    filters=16, #卷积核
    kernel_size=2,
    strides=1,
    padding='same',     # Padding method
    #data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
   # data_format='channels_first',
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 2, strides=1, padding='same'))#, data_format='channels_first'
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same'))#, data_format='channels_first'

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(2))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=[auc])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=20, batch_size=128,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)

print('\ntest accuracy: ', accuracy)

df = pd.read_csv('E:\\G-1149\\trafficCongestion\\训练数据\\测试数据\\test.csv', header=None)
data_pre = np.array(df).astype(float)
data_pre = data_pre.reshape(-1, 1,32, 2)/3
pre = model.predict_classes(data_pre)
print(pre)
