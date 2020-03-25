import numpy as np
from cost_functions.cost_function import cost_function

class categorical_crossentropy(cost_function):
    
    def calculateCostFunction(self, predY, groundY):
            m = len(predY)
            return np.squeeze(- np.sum(np.multiply(groundY, np.log(predY)))) 