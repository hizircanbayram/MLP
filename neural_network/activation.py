'''
import numpy as np

class activation():
    
    def activation_func(self, Z):
        pass
    
    def activation_func_drv(self, Z):
        pass
    
    
class tanh(activation):
    
    
    def activation_func(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    
    def activation_func_drv(self, Z):
        return 1 - np.square(self._tanh(Z))
    
    
class sigmoid(activation):
    

    def activation_func(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def activation_func_drv(self, Z):
        return self._sigmoid(Z) * (1 - self._sigmoid(Z))
    
    
class relu(activation):

    def activation_func(self, Z):
        return np.maximum(Z, 0)
    
    def activation_func_drv(self, Z):
        return (Z > 0) * 1
'''
