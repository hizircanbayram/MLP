# Multivariate Logistic Regression implemented by Hizir Can Bayram.

import numpy as np

from measuring_metrics import MeasurementMetrics

class MultivariateLogisticRegression():
    
    class Label():
        
        def __init__(self, Y_train, id, feature_num):
            self._id = id
            self._labels = []
            self._thetas = np.zeros((feature_num, 1)) 
            self._label_objs = 0
            self._cost_vals = []
            self._fillLabels(id, Y_train)
            
        
        
        # Creates an array that consists of 1 and 0 where 1 represents the class whose theta values are computed
        # Param | id : id of the class whose theta parameters are computed
        # Param | Y_train : dependent variable of shape numpy array.        
        def _fillLabels(self, id, Y_train):
            for sample in Y_train:
                if sample == id:
                    self._labels.append(1)
                else:
                    self._labels.append(0)
            self._labels = np.array(self._labels)
            
            
            
        # Returns the id field
        # Return | self._id : id value
        def getId(self):
            return self._id


            

            
    def __init__(self, learning_rate, epoch_num, reg_rate):
        #MeasurementMetrics.__init__(self, epoch_num)
        self._theta = 0
        self._training_sample = 0
        self._learning_rate = learning_rate
        self._epoch_num = epoch_num
        self._reg_rate = reg_rate
        self._cost_vals = []
        self._name = "MultivariateLogisticRegression"
    
    

    # Trains the model and calculates the parameters of multivariate logistic regression model
    # given as parameter.
    # Param | X : independent variables of shape numpy array.
    # Param | Y : dependent variable of shape numpy array.
    # Return | self._theta : theta values for per class    
    def train(self, X, Y):
       self._one_vs_all(X, Y, self._reg_rate)    
       


    # Returns label names in Y_train
    # Param | Y_train : dependent variable of shape numpy array.
    # Return | label_names : Name of labels    
    def _getLabelNames(self, Y_train):
        label_names = []
        for sample in Y_train:
            if sample not in label_names:
                label_names.append(sample)
        return label_names
        
    
        
    # Calculates theta values for every class and saves them into_label_objs
    # Param | X_train : independent variables of shape numpy array.
    # Param | Y_train : dependent variable of shape numpy array.
    def _one_vs_all(self, X_train, Y_train, reg_rate):
        label_names = self._getLabelNames(Y_train)
        self._label_objs = [self.Label(Y_train, label_names[i], X_train.shape[1]) for i in range(len(label_names))]    
        i = 0
        
        for obj in self._label_objs:
            obj._thetas = self._train(X_train, obj._labels, reg_rate)
            # for plotting cost function
            if i != (len(self._label_objs) - 1):
                self._cost_vals = []
                i += 1
            
            
            
    # Operates sigmoid function with the given parameter.
    # Param | power : thetas * transpose(X)
    # Return | : hypothesis function
    def _sigmoidFunction(self, power):
        power = -1 * power
        return 1 / (1 + (np.exp(np.array(power, dtype=np.int32))))
    
    
    
    # Trains the model and calculates the parameters of multivariate logistic regression model
    # given as parameter for per class.
    # Param | X : independent variables of shape numpy array.
    # Param | Y : dependent variable of shape numpy array.
    # Return | self._theta : theta values for per class
    def _train(self, X, Y, reg_rate):
        Y = Y.reshape((Y.size, 1))
        if len(X.shape) == 1:
            X = X.reshape((X.size, 1)) 
        bias = np.ones([X.shape[0], 1])
        X = np.concatenate((X, bias), 1)
        self._theta = np.zeros([X.shape[1], 1])
        self._training_sample = X.shape[0]
        
        for i in range (self._epoch_num):
            cost_val = self._calculateCost(X, Y)
            self._cost_vals.append(cost_val)
            hypothesis = self._sigmoidFunction(X.dot(self._theta))  
            difference = np.subtract(hypothesis, Y) # hypothesis function - y values (for all training sample in the dataset, leading a vector of size m where m is the training sample in the dataset)
            cost_func = np.transpose(X).dot(difference)
            gradient = (self._learning_rate / self._training_sample) * cost_func
            if self._reg_rate != None:
                reg_factor = (1 - (self._learning_rate * self._reg_rate) / self._training_sample)
                self._theta = np.subtract(reg_factor * self._theta, gradient)
            else:
                self._theta = np.subtract(self._theta, gradient)

        return self._theta
    
    
    
    def _calculateCost(self, X, Y):
        first_term = np.transpose(np.log(self._sigmoidFunction(X.dot(self._theta)))).dot(Y)
        ones = np.ones((len(Y), 1))
        second_term = np.transpose(np.log(np.subtract(ones, self._sigmoidFunction(X.dot(self._theta))))).dot(np.subtract(ones, Y))
        return (first_term[0][0] + second_term[0][0]) * (-1 / self._training_sample)
    
    
    
    # Predicts the dependent variable based on the given independent variable as parameter.
    # Param | X : independent variable sample of shape python list.
    # Return | : prediction of shape a number.
    def predict(self, X):
        X = np.array(X)
        bias = np.ones([1]).reshape((1,1))
        if len(X.shape) == 1:
            X = X.reshape((X.size, 1)) 
        X = np.concatenate((X, bias), 1)
        
        preds = []
        for obj in self._label_objs:
            #print('Theta : ', len(obj._thetas))
            preds.append(X.dot(obj._thetas))

        return self._label_objs[preds.index(max(preds))].getId()



    def getCostVals(self):
        return self._cost_vals
       
    def getName(self):
        return self._name
    
    def getRegRate(self):
        return self._reg_rate
    
    def getLearningRate(self):
        return self._learning_rate
    
    def getEpochNum(self):
        return self._epoch_num
    
    def getFinalCostValue(self):
        return self._cost_vals[self._epoch_num - 1]