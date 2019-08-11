# Development of measuring metric tools implemented by Hizir Can Bayram.

import matplotlib.pyplot as plt

class MeasurementMetrics():
    
    def __init__(self, iter_num):
        self._cost_vals = []
        self._iter_num = iter_num
        
        
        
    # Plots a graph showing the changes in cost function based on number of iterations through the training. Its x-axis is limited with the number given as parameter.
    # Param | x_axis : Length of x-axis of the graph
    def plotCostFunction(self, x_axis):
        iters = [None] * self._iter_num
        for i in range(x_axis):
            iters[i] = i
        
        plt.plot(iters, self._cost_vals)
        
        
        
    # Gives the value of final cost function based on squared error cost function.
    # Return | : value of cost function created with latest parameters
    def getFinalCost(self):
        return self._cost_vals[self._iter_num - 1]
