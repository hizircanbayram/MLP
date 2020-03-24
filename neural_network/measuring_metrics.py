import numpy as np

def measureAccuracy(Y_pred, Y_test):
    if len(Y_pred) != len(Y_test):
        print('len(Y_pred):', len(Y_pred), 'len(Y_test):', len(Y_test), 'correct it.')
        return
    equal_no = 0
    for i in range(len(Y_pred)):
        if np.array_equal(Y_pred[i], Y_test[i]):
            equal_no += 1
    return "{0:,.3f}".format(100 * equal_no / len(Y_test))


def plotLearningGraph(errors):
    import matplotlib.pyplot as plt
    plt.plot(errors)
    plt.ylabel('Training error')
    plt.xlabel('Epoch no')
    plt.show()



