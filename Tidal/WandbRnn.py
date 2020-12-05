from __future__ import print_function

import wandb
from wandb.keras import WandbCallback
wandb.init(project="pq")

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from Tidal import dao

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

np.random.seed(1337)  # for reproducibility

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, LSTM
from keras.optimizers import Adam

from sklearn.metrics import roc_auc_score

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

TIME_STEPS = 2     # same as the height of the image
INPUT_SIZE = 16     # same as the width of the image
BATCH_SIZE = 35
BATCH_INDEX = 0
OUTPUT_SIZE = 2
CELL_SIZE = 80
LR = 0.001
EPOCHS = 100

df = pd.read_csv('E:\\G-1149\\trafficCongestion\\训练数据\\4小时文件\\trainData\\res\\resSam.csv', header=None)
data_sample = np.array(df).astype(float)
df = pd.read_csv('E:\\G-1149\\trafficCongestion\\训练数据\\4小时文件\\trainData\\res\\resLabel.csv', header=None)
data_label = np.array(df).astype(float)


X_train, X_test, y_train, y_test = train_test_split(data_sample,data_label,test_size=0.3, random_state=0)
print(X_train.size)
print(y_train.size)

# data pre-processing

X_train = X_train.reshape(-1, 2, 16)/3      # normalize
X_test = X_test.reshape(-1, 2, 16)/3      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)

# build RNN model
model = Sequential()

# RNN cell
model.add(LSTM(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    #output_dim=CELL_SIZE,
    unroll=True,
    units=CELL_SIZE
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              #metrics=[auc])
              metrics=['accuracy'])

# training
history = model.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)


y_pred = model.predict_classes(X_test)
y_test_new = []
tp = 0
tn = 0
fp = 0
fn = 0
index = 0
for i in y_test:
    if(i[1] == 1):
        y_test_new.append(1)
        if(y_pred[index] == 1):
            tp += 1
        else:
            fn += 1
    else:
        y_test_new.append(0)
        if(y_pred[index] == 1):
            fp += 1
        else:
            tn += 1
    index += 1

print('tp ', tp, ' fn ', fn, ' fp ', fp, ' tn ', tn)
print('accuracy: ', accuracy_score(y_test_new, y_pred), ' ', (tp+tn)/(tp+tn+fp+fn))
print('precision: ', precision_score(y_test_new, y_pred, average='micro'),' ', tp/(tp+fp))
print('recall: ', recall_score(y_test_new, y_pred, average='micro'),' ', tp/(tp+fn))
print('f1: ', f1_score(y_test_new, y_pred, average='micro'),' ', 2*tp/(2*tp+fp+fn))

df = pd.read_csv('data/test_4.csv', header=None)
data_pre = np.array(df).astype(float)
data_pre = data_pre.reshape(-1, 2, 16)/3
pre = model.predict_classes(data_pre)
print(pre)
dao.score(pre)

print(model.summary())

import os
# Save model to wandb
model.save(os.path.join(wandb.run.dir, "model.h5"))