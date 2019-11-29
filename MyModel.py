import pickle
import PreProcessing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Activation
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import time
import os

EPOCHS = 300
BATCH_SIZE = 64
NAME = f"PRED-{int(time.time())}"

model = Sequential()
model.add(LSTM(units = 128, return_sequences = True, input_shape = (PreProcessing.trainX.shape[1],PreProcessing.trainX.shape[2])))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(units = 128, return_sequences = True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(units = 128))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(units = 32))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics= ['accuracy'])

tensorboard = TensorBoard(log_dir = os.path.join(
    "logs",
    "fit",
    NAME,
))

filepath = "RNN_Final-{epoch:02d}-{accuracy:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = model.fit(
    PreProcessing.trainX, PreProcessing.trainY,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[tensorboard, checkpoint],
    validation_data=(PreProcessing.valX, PreProcessing.valY),
    shuffle=False
)

pickle.dump(model, open("Model7", 'wb'))

score = model.evaluate(PreProcessing.testX, PreProcessing.testY, verbose=0)
print('Test loss:', score[0])
