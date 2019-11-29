import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.random.seed(3)
step_size = 30
scaler = MinMaxScaler(feature_range=(0, 1))

def time_series(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)

dataset = pd.read_csv('Amazon.us.txt', usecols=[1,2,3,4])
dataset = dataset.reindex(index = dataset.index[::-1])

features = dataset.mean(axis = 1)
features = np.reshape(features.values, (len(features),1))
features = scaler.fit_transform(features)

train, test = train_test_split(features, test_size=0.2, shuffle=False, random_state=3)
train, val = train_test_split(train, test_size=0.15, shuffle=False, random_state=3)

trainX, trainY = time_series(train,1)
testX, testY = time_series(test, 1)
valX, valY = time_series(val, 1)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
valX = np.reshape(valX, (valX.shape[0], 1, valX.shape[1]))
