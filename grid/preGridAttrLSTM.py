from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from Tidal import dao
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import csv

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, LSTM, Dropout
from keras.optimizers import Adam

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

# df = pd.read_csv('E:\\G-1149\\trafficCon长短时记忆神经gestion\\训练数据\\trainData\\Sample15_1.csv', header=None)
# data_sample = np.array(df).astype(float)
# df = pd.read_csv('E:\\G-1149\\trafficCongestion\\训练数据\\trainData\\label.csv', header=None)
# data_label = np.array(df).astype(float)


X_train, X_test, y_train, y_test = train_test_split(data_sample,data_label,test_size=0.3, random_state=0)
print(X_train.size)
print(y_train.size)

# data pre-processing

X_train = X_train.reshape(-1, 2, 16)/3      # normalize
X_test = X_test.reshape(-1, 2, 16)/3      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)


model = Sequential()

model.add(LSTM(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    #output_dim=CELL_SIZE,
    unroll=True,
    units=CELL_SIZE
))
#model.add(LSTM(32))
#model.add(Dropout(0.2))

model.add(Dense(units = OUTPUT_SIZE))
model.add(Activation('softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)

acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

plt.title('Loss')
#plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, loss, 'blue', label='Validation loss')
plt.xlabel('epochs');
plt.legend()
plt.show()

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

# print('tp ', tp, ' fn ', fn, ' fp ', fp, ' tn ', tn)
print('accuracy: ', accuracy_score(y_test_new, y_pred), ' ', (tp+tn)/(tp+tn+fp+fn))
# print('precision: ', precision_score(y_test_new, y_pred, average='micro'),' ', tp/(tp+fp))
# print('recall: ', recall_score(y_test_new, y_pred, average='micro'),' ', tp/(tp+fn))
# print('f1: ', f1_score(y_test_new, y_pred, average='micro'),' ', 2*tp/(2*tp+fp+fn))

# 可视化
# plt.plot(y_test, color = 'black', label = 'SZ000001 Price')
# plt.plot(y_pred, color = 'green', label = 'Predicted SZ000001 Price')
# plt.title('SZ000001 Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('SZ000001 Price')
# plt.legend()
# plt.show()


print(model.summary())


gridLinkPeerPath = 'E:/G-1149/trafficCongestion/网格化/gridLinkPeer_14.csv'
gridTidalPath = 'E:/G-1149/trafficCongestion/网格化/tidal/gridTidal_rnn_lstm14.csv'
linkStatusPath = "E:/G-1149/trafficCongestion/网格化/linkStatus_14_完整.csv"
test = '../Tidal/data/test_4.csv'

#gridTidalPath_1 = 'E:/G-1149/trafficCongestion/网格化/tidal/gridTidal_rnn_14_1.csv'

statusDict = {}
with open(linkStatusPath, 'r') as file:
    reader = csv.reader(file)
    for r in reader:    # r是一个list
        statusDict[r[0]] = r[1:]

f = open(gridTidalPath, 'w', encoding='utf-8', newline ='') #newline解决空行问题
csv_writer = csv.writer(f)
# f1 = open(gridTidalPath_1, 'w', encoding='utf-8', newline ='') #newline解决空行问题
# csv_writer1 = csv.writer(f1)

with open(gridLinkPeerPath, 'r') as file:
    reader = csv.reader(file)
    for r in reader:    # r是一个list
        if(len(r) == 1):
            continue
        linkPeer = r[1:]
        peerList = []
        for i in range(len(linkPeer)):
            if(linkPeer[i] not in statusDict.keys()):
                break
            if(i % 2 != 0):
                pre = np.array(linkStatus + statusDict[linkPeer[i]]).astype(float)
                pre = pre.reshape(-1, 2, 16) / 3
                pre = model.predict_classes(pre)
                if(pre[0] == 1):
                    #csv_writer1.writerow([r[0], linkPeer[i-1], linkPeer[i]])
                    peerList.append(linkPeer[i-1])
                    peerList.append(linkPeer[i])
            else:
                linkStatus = statusDict[linkPeer[i]]
        if(len(peerList) > 1):
            peerList.insert(0, r[0])
            csv_writer.writerow(peerList)

f.close()
#f1.close()