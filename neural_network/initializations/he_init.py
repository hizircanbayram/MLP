from initializations.initializer import initializer
import numpy as np

class he_init(initializer):
    '''
    This initialization method is usually used with relu/leaky relu function.
    '''
    def initializeWeights(self, x, y):
        return np.random.randn(x, y) * np.sqrt(2 / y)