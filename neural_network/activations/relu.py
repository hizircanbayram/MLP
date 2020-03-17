from activation import activation
import numpy as np

class relu(activation):

    def activation_func(self, Z):
        return np.maximum(Z, 0)
    
    def activation_func_drv(self, Z):
        return (Z > 0) * 1

