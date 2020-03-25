import numpy as np
from cost_functions.cost_function import cost_function

class binary_crossentropy(cost_function):
    
    def calculateCostFunction(self, predY, groundY):
            m = len(predY)
            return np.squeeze((- 1 / m) * np.sum(np.multiply(groundY, np.log(predY)) + 
                                      np.multiply((1 - groundY), np.log(1 - predY))))   