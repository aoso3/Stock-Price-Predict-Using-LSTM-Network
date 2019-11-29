import PreProcessing
import pickle
import numpy as np
import Visualization

model = pickle.load(open("Model7", 'rb'))

score = model.evaluate(PreProcessing.testX, PreProcessing.testY, verbose=0)
print('Test loss:', score[0])

Original_data = PreProcessing.scaler.inverse_transform(PreProcessing.features)
train = PreProcessing.scaler.inverse_transform(PreProcessing.train)
validation = PreProcessing.scaler.inverse_transform(PreProcessing.val)
testPredict = PreProcessing.scaler.inverse_transform(model.predict(PreProcessing.testX))

validationPlot = np.empty_like(PreProcessing.features)
validationPlot[:, :] = np.nan
validationPlot[len(train) : len(PreProcessing.features) -  len(PreProcessing.test) , :] = validation

testPredictPlot = np.empty_like(PreProcessing.features)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(PreProcessing.features) -  len(PreProcessing.test) + 1 :len(PreProcessing.features) - 1, :] = testPredict

Visualization.Plot(Original_data,train,validationPlot,testPredictPlot)
