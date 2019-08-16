# Multivariate Logistic Regression implemented by Hizir Can Bayram.

import numpy as np

from measuring_metrics import MeasurementMetrics

class MultivariateLogisticRegression(MeasurementMetrics):
    
    class Label():
        
        def __init__(self, Y_train, id, feature_num):
            self._id = id
            self._labels = []
            self._thetas = np.zeros((feature_num, 1)) # duruma gore list olarak da tanimlanabilir
            self._label_objs = 0
            self._fillLabels(id, Y_train)
            
            
        def _fillLabels(self, id, Y_train):
            for sample in Y_train:
                if sample == id:
                    self._labels.append(1)
                else:
                    self._labels.append(0)
            self._labels = np.array(self._labels)
            
        def getId(self):
            return self._id


            
            
    def __init__(self, learning_rate, epoch_num):
        MeasurementMetrics.__init__(self, epoch_num)
        self._theta = 0
        self._training_sample = 0
        self._learning_rate = learning_rate
        self._epoch_num = epoch_num
 

    
    def train(self, X, Y):
       return self._one_vs_all(X, Y)    
       


    def _getLabelNames(self, Y_train):
        label_names = []
        for sample in Y_train:
            if sample not in label_names:
                label_names.append(sample)
        return label_names
        
        
    
    def _one_vs_all(self, X_train, Y_train):
        label_names = self._getLabelNames(Y_train)
        self._label_objs = [self.Label(Y_train, label_names[i], X_train.shape[1]) for i in range(len(label_names))]    
        for obj in self._label_objs:
            obj._thetas = self._train(X_train, obj._labels)
        return self._label_objs   
        
    
    
    # Operates sigmoid function with the given parameter.
    # Param | power : thetas * transpose(X)
    # Return | : hypothesis function
    def _sigmoidFunction(self, power):
        power = -1 * power
        return 1 / (1 + (np.exp(np.array(power, dtype=np.int32))))
    
    
    
    # Trains the model and calculates the parameters of multivariate logistic regression model
    # given as parameter.
    # Param | X : independent variables of shape numpy array.
    # Param | Y : dependent variable of shape numpy array.
    def _train(self, X, Y):
        Y = Y.reshape((Y.size, 1))
        if len(X.shape) == 1:
            X = X.reshape((X.size, 1)) 
        bias = np.ones([X.shape[0], 1])
        X = np.concatenate((X, bias), 1)
        self._theta = np.zeros([X.shape[1], 1])
        self._training_sample = X.shape[0]

        for i in range (self._epoch_num):
            hypothesis = self._sigmoidFunction(X.dot(self._theta))  
            difference = np.subtract(hypothesis, Y) # hypothesis function - y values (for all training sample in the dataset, leading a vector of size m where m is the training sample in the dataset)
            cost_val = (np.sum(difference) ** 2) / self._training_sample
            self._cost_vals.append(cost_val)
            cost_func = np.transpose(X).dot(difference)
            gradient = (self._learning_rate / self._training_sample) * cost_func
            self._theta = np.subtract(self._theta, gradient)
        #print('thetas : ', self._theta)
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
        
        max_rate = 0
        max_id = None
        
        preds = []
        for obj in self._label_objs:
            preds.append(X.dot(obj._thetas))

        return self._label_objs[preds.index(max(preds))].getId()


