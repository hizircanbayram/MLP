import numpy as np
from optimizer import optimizer

class momentum(optimizer):
    
    def __init__(self, learning_rate=0.0001, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.inited = False
        self.vdW = None
        self.vdb = None
        
    
    def _backwardPropagation(self, predY, groundY, nn_obj):
        new_weights = []
        new_bias = []
        dAL = - (np.divide(groundY, predY) - np.divide(1 - groundY, 1 - predY))
        
        if self.inited == False:
            self.vdW = [None] * nn_obj._layer_no
            self.vdb = [None] * nn_obj._layer_no
            for i in reversed(range(nn_obj._layer_no)):
                self.vdW[i] = np.zeros(nn_obj.weights[i].shape)
                self.vdb[i] = np.zeros(nn_obj.bias[i].shape)
            self.inited = True
            
        for i in reversed(range(nn_obj._layer_no)):
            ZL = nn_obj.Zs[i]
            if i == nn_obj._layer_no - 1 and nn_obj.act_funcs[i].name == 'softmax':
                '''
                We don't use dAL to compute dZL for the final layer when it comes to 
                softmax. We directly compute dZL instead as shown in the below. Since
                dZL is computed anyway, dAL for the previous layer can move on where it
                left off(dAL = np.dot(nn_obj.weights[i].T, dZL))
                '''
                dZL = predY - groundY
            else:
                dZL = np.multiply(dAL, nn_obj.act_funcs[i].activation_func_drv(ZL))
            AL1 = nn_obj.As[i] # A[l - 1]
            m = len(predY)
            dWL = (1 / m) * np.dot(dZL, AL1.T)
            dbL = (1 / m) * np.sum(dZL, axis=1, keepdims=True)
            self.vdW[i] = self.beta * self.vdW[i] + (1 - self.beta) * dWL
            self.vdb[i] = self.beta * self.vdb[i] + (1 - self.beta) * dbL
            new_weights.append(nn_obj.weights[i] - self.learning_rate * self.vdW[i])
            new_bias.append(nn_obj.bias[i] - self.learning_rate * self.vdb[i])
            dAL = np.dot(nn_obj.weights[i].T, dZL)
        return list(reversed(new_weights)), list(reversed(new_bias))