import numpy as np
from initializations.initializer import initializer

class standart_normal(initializer):
    
    def initializeWeights(self, x, y):
        return np.random.randn(x, y)