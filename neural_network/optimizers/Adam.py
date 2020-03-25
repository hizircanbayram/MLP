import math
import numpy as np
from optimizer import optimizer

class Adam(optimizer):
    
    def __init__(self, learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon 
        self.inited = False
        self.vdW = None
        self.vdb = None
        self.sdW = None
        self.sdb = None
        
    
    def _backwardPropagation(self, predY, groundY, nn_obj):
        new_weights = []
        new_bias = []
        dAL = - (np.divide(groundY, predY) - np.divide(1 - groundY, 1 - predY))
        
        if self.inited == False:
            self.vdW = [None] * nn_obj._layer_no
            self.vdb = [None] * nn_obj._layer_no
            self.sdW = [None] * nn_obj._layer_no
            self.sdb = [None] * nn_obj._layer_no
            for i in reversed(range(nn_obj._layer_no)):
                self.vdW[i] = np.zeros(nn_obj.weights[i].shape)
                self.vdb[i] = np.zeros(nn_obj.bias[i].shape)
                self.sdW[i] = np.zeros(nn_obj.weights[i].shape)
                self.sdb[i] = np.zeros(nn_obj.bias[i].shape)
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
            self.vdW[i] = self.beta1 * self.vdW[i] + (1 - self.beta1) * dWL
            self.vdb[i] = self.beta1 * self.vdb[i] + (1 - self.beta1) * dbL
            self.sdW[i] = self.beta2 * self.sdW[i] + (1 - self.beta2) * np.square(dWL)
            self.sdb[i] = self.beta2 * self.sdb[i] + (1 - self.beta2) * np.square(dbL)
            V_cor_dW = self.vdW[i] / (1 - math.pow(self.beta1, nn_obj.current_epoch))
            V_cor_db = self.vdb[i] / (1 - math.pow(self.beta1, nn_obj.current_epoch))
            S_cor_dW = self.sdW[i] / (1 - math.pow(self.beta2, nn_obj.current_epoch))
            S_cor_db = self.sdb[i] / (1 - math.pow(self.beta2, nn_obj.current_epoch))
            new_weights.append(nn_obj.weights[i] - 
                        (self.learning_rate * (V_cor_dW / (np.sqrt(S_cor_dW) + self.epsilon))))
            new_bias.append(nn_obj.bias[i] - 
                        (self.learning_rate * (V_cor_db / (np.sqrt(S_cor_db) + self.epsilon))))
            dAL = np.dot(nn_obj.weights[i].T, dZL)
        return list(reversed(new_weights)), list(reversed(new_bias))