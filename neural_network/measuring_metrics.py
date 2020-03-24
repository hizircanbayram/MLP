import numpy as np

def measureAccuracy(Y_pred, Y_test):
    if len(Y_pred) != len(Y_test):
        print('len(Y_pred):', len(Y_pred), 'len(Y_test):', len(Y_test), 'correct it.')
        return
    return 100 - (100 * np.sum(np.absolute(Y_pred - Y_test))) / len(Y_test)


def plotLearningGraph(errors):
    import matplotlib.pyplot as plt
    plt.plot(errors)
    plt.ylabel('Training error')
    plt.xlabel('Epoch no')
    plt.show()



