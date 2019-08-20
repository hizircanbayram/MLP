# Multivariate Linear Regression implemented by Hizir Can Bayram.

import sys
import numpy as np


class MultivariateLinearRegression():
    

    # Param | reg_rate : if the model is regularized linear regression, regularization rate        
    def __init__(self, learning_rate, learning_algorithm, epoch_num, reg_rate = None):
        #MeasurementMetrics.__init__(self, epoch_num)
        self._learning_rate = learning_rate
        self._learning_algoritm = learning_algorithm
        self._epoch_num = epoch_num
        self._theta = 0
        self._training_sample = 0
        self._reg_rate = reg_rate
        self._cost_vals = []
        self._log = []
        self._name = "MultivariateLinearRegression"
    
    
    
    # Trains the model and calculates the parameters of multivariate linear regression model based on the learning algorithm
    # given as parameter.
    # Param | X : independent variables of shape numpy array.
    # Param | Y : dependent variable of shape numpy array.
    def train(self, X, Y):
        if self._learning_algoritm == 'Gradient Descent':
            self._trainGradientDescent(X, Y, self._reg_rate)
        elif self._learning_algoritm == 'Normal Equation':
            self._trainNormalEquation(X, Y, self._reg_rate)
        else:
            print('No such a learning algorithm. Check the learning algorithm\'s name given as parameter. It should be either \'Gradient Descent\' or \'Normal Equation\'')
            sys.exit()



    # Trains the model with gradient descent optimization algorithm.
    # Param | X : independent variables of shape numpy array.
    # Param | Y : dependent variable of shape numpy array.   
    def _trainGradientDescent(self, X, Y, reg_rate):
        Y = Y.reshape((Y.size, 1))
        if len(X.shape) == 1:
            X = X.reshape((X.size, 1)) 
        bias = np.ones([X.shape[0], 1])
        X = np.concatenate((X, bias), 1)
        
        self._theta = np.zeros([X.shape[1], 1])
        self._training_sample = X.shape[0]
        loading_rate = self._epoch_num // 10
        percent_rate = 0
        print('%', end=' ')
        for i in range (self._epoch_num):
            hypothesis = X.dot(self._theta)
            difference = np.subtract(hypothesis, Y) # hypothesis function - y values (for all training sample in the dataset, leading a vector of size m where m is the training sample in the dataset)
            cost_func = np.transpose(X).dot(difference)
            gradient = (self._learning_rate / self._training_sample) * cost_func
            if self._reg_rate != None:
                # cost value calculation
                linear_cost_val = (np.sum(difference) ** 2) / self._training_sample
                reg_cost_val = reg_rate * np.sum(self._theta ** 2)
                cost_val = linear_cost_val + reg_cost_val
                # theta parameters updating
                reg_factor = (1 - (self._learning_rate * self._reg_rate) / self._training_sample)
                self._theta = np.subtract(reg_factor * self._theta, gradient)
            else:
                cost_val = (np.sum(difference) ** 2) / self._training_sample  # cost value calculation                              
                self._theta = np.subtract(self._theta, gradient) # theta parameters updating
            self._cost_vals.append(cost_val) # cost value updating
        # piece of instructions to show how long it takes to finish training
            if loading_rate != 0 and i % loading_rate == 0:
                percent_rate += 1
                print(percent_rate * 10, end=' ')
            self._log.append(self._theta)
        if loading_rate == 0:
            print(100)        
        print()

        
        
    # Trains the model with normal equation optimization algorithm.
    # Param | X : independent variables of shape numpy array.
    # Param | Y : dependent variable of shape numpy array. 
    def _trainNormalEquation(self, X, Y):
        print('it will be implemented')

    
    
    # Predicts the dependent variable based on the given independent variable as parameter.
    # Param | X : independent variable sample of shape python list.
    # Return | : prediction of shape a number.
    def predict(self, X):
        X = np.array(X)
        bias = np.ones([1]).reshape((1,1))
        if len(X.shape) == 1:
            X = X.reshape((X.size, 1)) 
        X = np.concatenate((X, bias), 1)

        return X.dot(self._theta)[0][0]



    def getFinalCostValue(self):
        return self._cost_vals[self._epoch_num - 1]
    
    def getName(self):
        return self._name
    
    def getRegRate(self):
        return self._reg_rate
    
    def getLearningRate(self):
        return self._learning_rate
    
    def getEpochNum(self):
        return self._epoch_num
    