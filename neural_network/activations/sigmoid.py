from activation import activation
import numpy as np

class sigmoid(activation):
 
    def __init__(self):
        self.name = 'sigmoid' 

    def activation_func(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def activation_func_drv(self, Z):
        return self.activation_func(Z) * (1 - self.activation_func(Z))