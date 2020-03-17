from activation import activation
import numpy as np

class tanh(activation):
    
    
    def activation_func(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    
    def activation_func_drv(self, Z):
        return 1 - np.square(self.activation_func(Z))