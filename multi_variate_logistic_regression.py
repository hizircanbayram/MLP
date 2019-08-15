# Multivariate Logistic Regression implemented by Hizir Can Bayram.

import numpy as np

from measuring_metrics import MeasurementMetrics

class MultivariateLogisticRegression(MeasurementMetrics):
    

    
    def __init__(self, learning_rate, epoch_num):
        MeasurementMetrics.__init__(self, epoch_num)
        self._theta = 0
        self._training_sample = 0
        self._learning_rate = learning_rate
        self._epoch_num = epoch_num
        
        
        
    def _sigmoidFunction(self, power):
        power = -1 * power
        return 1 / (1 + (np.exp(power)))

    
    
    # Trains the model and calculates the parameters of multivariate logistic regression model
    # given as parameter.
    # Param | X : independent variables of shape numpy array.
    # Param | Y : dependent variable of shape numpy array.
    def train(self, X, Y):
        Y = Y.reshape((1, Y.size))
        if len(X.shape) == 1:
            X = X.reshape((X.size, 1)) 
        bias = np.ones([X.shape[0], 1])
        X = np.concatenate((X, bias), 1)
        self._theta = np.zeros([1, X.shape[1]])
        self._training_sample = X.shape[0]
        
        
        for i in range (self._epoch_num):
            hypothesis = self._sigmoidFunction(self._theta.dot(np.transpose(X)))  
            difference = np.subtract(hypothesis, Y) # hypothesis function - y values (for all training sample in the dataset, leading a vector of size m where m is the training sample in the dataset)
            cost_val = (np.sum(difference) ** 2) / self._training_sample
            self._cost_vals.append(cost_val)
            cost_func = difference.dot(X)
            gradient = (self._learning_rate / self._training_sample) * cost_func
            self._theta = np.subtract(self._theta, gradient)
               
        return self._theta
    
    
    # Predicts the dependent variable based on the given independent variable as parameter.
    # Param | X : independent variable sample of shape python list.
    # Return | : prediction of shape a number.
    def predict(self, X):
        X = np.array(X)
        bias = np.ones([1]).reshape((1,1))
        if len(X.shape) == 1:
            X = X.reshape((X.size, 1)) 
        
        X = np.concatenate((X, bias), 1)
        if X.dot(np.transpose(self._theta))[0] >= 0.5:
            return 1
        else:
            return 0

