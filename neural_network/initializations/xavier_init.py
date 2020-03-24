from initializations.initializer import initializer
import numpy as np

class xavier_init(initializer):
    '''
    This initialization method is usually used with tanh function.
    '''
    def initializeWeights(self, x, y):
        return np.random.randn(x, y) * np.sqrt(1 / y)