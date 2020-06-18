import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv('C:\\Users\\98259\\Desktop\\RNN\\LSTM_learn\\international-airline-passengers.csv', sep=',')
# df = df.set_index('time')
# df['passengers'].plot()
# plt.show()

file_name = 'C:\\Users\\98259\\Desktop\\RNN\\LSTM_learn\\international-airline-passengers.csv'
df = pd.read_csv(file_name, sep=',', usecols=[1])
data_all = np.array(df).ravel().astype(float)
#print(data_all)
data = []
sequence_length = 10
for i in range(len(data_all) - sequence_length - 1):
    data.append(data_all[i: i + sequence_length + 1])
reshaped_data = np.array(data).astype('float64')
#print(reshaped_data)

split = 0.8
#np.random.shuffle(reshaped_data)
x = reshaped_data[:, :-1]
y = reshaped_data[:, -1]
split_boundary = int(reshaped_data.shape[0] * split)
train_x = x[: split_boundary]
test_x = x[split_boundary:]

train_y = y[: split_boundary]
test_y = y[split_boundary:]
print(train_x)

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
print(train_x)