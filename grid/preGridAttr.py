import csv

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

np.random.seed(1337)  # for reproducibility

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

TIME_STEPS = 16     # same as the height of the image
INPUT_SIZE = 2     # same as the width of the image
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 2
CELL_SIZE = 50
LR = 0.001

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
#(X_train, y_train), (X_test, y_test) = mnist.load_data()


df = pd.read_csv('E:\\G-1149\\trafficCongestion\\训练数据\\4小时文件\\trainData\\res\\resSam.csv', header=None)
data_sample = np.array(df).astype(float)
df = pd.read_csv('E:\\G-1149\\trafficCongestion\\训练数据\\4小时文件\\trainData\\res\\resLabel.csv', header=None)
data_label = np.array(df).astype(float)

X_train, X_test, y_train, y_test = train_test_split(data_sample,data_label,test_size=0.3, random_state=0)
print(X_train.size)
print(y_train.size)

# data pre-processing

X_train = X_train.reshape(-1, 16, 2)/3      # normalize
X_test = X_test.reshape(-1, 16, 2)/3      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)

# build RNN model
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    unroll=True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
for step in range(4001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)


gridLinkPeerPath = 'E:/G-1149/trafficCongestion/网格化/gridLinkPeer_13.csv'
gridTidalPath = 'E:/G-1149/trafficCongestion/网格化/tidal/gridTidal_rnn_13.csv'
linkStatusPath = "E:/G-1149/trafficCongestion/网格化/linkStatus_13_完整.csv"
test = '../Tidal/data/test_4.csv'

statusDict = {}
with open(linkStatusPath, 'r') as file:
    reader = csv.reader(file)
    for r in reader:    # r是一个list
        statusDict[r[0]] = r[1:]

f = open(gridTidalPath, 'w', encoding='utf-8', newline ='') #newline解决空行问题
csv_writer = csv.writer(f)

with open(gridLinkPeerPath, 'r') as file:
    reader = csv.reader(file)
    for r in reader:    # r是一个list
        if(len(r) == 1):
            continue
        linkPeer = r[1:]
        for i in range(len(linkPeer)):
            if(linkPeer[i] not in statusDict.keys()):
                break
            if(i % 2 != 0):
                # print(r[0])
                # print(linkStatus + statusDict[linkPeer[i]])
                pre = np.array(linkStatus + statusDict[linkPeer[i]]).astype(float)
                pre = pre.reshape(-1, 16, 2) / 3
                pre = model.predict_classes(pre)
                csv_writer.writerow([r[0], pre[0]])     #网格编号，预测值
            else:
                linkStatus = statusDict[linkPeer[i]]

f.close()