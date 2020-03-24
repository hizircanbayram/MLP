from activation import activation
import numpy as np

class softmax(activation):
    
    def __init__(self):
        self.name = 'softmax'

    def activation_func(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    
    def activation_func_drv(self, Z):
        pass

