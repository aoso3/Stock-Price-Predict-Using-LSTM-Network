import matplotlib.pyplot as plt

def Plot(data,trainPlot,validationPlot,testPredictPlot):
    plt.plot(data, 'g', label='Original Dataset')
    plt.plot(trainPlot, 'y', label='Train data')
    plt.plot(validationPlot, 'r', label='Validation data')
    plt.plot(testPredictPlot, 'b', label='Predicted stock price')
    plt.legend(loc='upper right')
    plt.xlabel('Time/Days')
    plt.ylabel('Stock Value')
    plt.show()