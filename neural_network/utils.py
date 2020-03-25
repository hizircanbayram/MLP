import numpy as np

def convert_one_hot(Y):
    Y = Y.T
    Y_onehot = np.zeros((Y.size, Y.max()+1), dtype='int32')
    Y_onehot[np.arange(Y.size),Y] = 1
    return Y_onehot