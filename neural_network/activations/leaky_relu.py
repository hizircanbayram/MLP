import numpy as np

class leaky_relu():
    
    def __init__(self, alpha=0.01):
        self._name = 'leaky_relu'
        self.alpha = alpha
        
    def activation_func(self, Z):
        return np.maximum(Z, self.alpha*Z) 
    
    def activation_func_drv(self, Z):
        drv_relu = np.ones_like(Z)
        drv_relu[Z < 0] = self.alpha
        return drv_relu