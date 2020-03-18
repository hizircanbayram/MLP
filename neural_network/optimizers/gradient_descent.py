import numpy as np
from optimizer import optimizer

class gd(optimizer):
    
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate
        
    
    def _backwardPropagation(self, predY, groundY, nn_obj):
        new_weights = []
        new_bias = []
        dAL = - (np.divide(groundY, predY) - np.divide(1 - groundY, 1 - predY))
        for i in reversed(range(nn_obj._layer_no)):
            ZL = nn_obj.Zs[i]
            dZL = np.multiply(dAL, nn_obj.act_funcs[i].activation_func_drv(ZL))
            AL1 = nn_obj.As[i] # A[l - 1]
            m = len(predY)
            dWL = (1 / m) * np.dot(dZL, AL1.T)
            dbL = (1 / m) * np.sum(dZL, axis=1, keepdims=True)
            new_weights.append(nn_obj.weights[i] - self.learning_rate * dWL)
            new_bias.append(nn_obj.bias[i] - self.learning_rate * dbL)
            dAL = np.dot(nn_obj.weights[i].T, dZL)
        return list(reversed(new_weights)), list(reversed(new_bias))