import numpy as np


class NeuralNetwork():

    # DO ANOTHER MODULE FOR INITIALIZATION
    # DO ANOTHER MODULE FOR OPTIMIZATION ALGORITHMS
    # ADD PREDICT FUNCTION

    def __init__(self):
        # layer parameters
        self.weights = []
        self.bias = []
        self.act_funcs = []
        self._layer_no = 0 # keeps track of number of layers in the NN
        # model parameters
        self.optimizer = None
        self.loss = None # loss(error) function's name
        self.epoch = 0
        # cache
        self.errors = []
        self.Zs = []
        self.As = []
        
        
    def createLayer(self, neuron_size, act_func, input_dim=None):
        if self._layer_no == 0 and input_dim == None:
            print('Input size has to be determined')
            return
        if self._layer_no != 0 and input_dim != None:
            print('Input size must not be determined')
            return
        # Either granted, self._init_cnt != 0 or input_dim != None. Not both
        if input_dim != None: # First layer
            created_weight = np.random.randn(neuron_size, input_dim)
        if self._layer_no != 0:
            second_dim = self.weights[self._layer_no - 1].shape[0]
            created_weight = np.random.randn(neuron_size, second_dim)
        self.weights.append(created_weight)
        self.bias.append(np.zeros((neuron_size, 1)))
        self.act_funcs.append(act_func)
        self._layer_no += 1
        
        
    def train(self, X, Y):
        for i in range(self.epoch):
            predY = self._forwardPropagation(X)
            self._calculateCostFunction(predY, Y)
            self._backwardPropagation(predY, Y)            
            self.As = []
            self.Zs = []

        
    def _forwardPropagation(self, X):
        A = X
        self.As.append(A)
        for i in range(self._layer_no):
            W = self.weights[i]
            Z = np.asarray(np.dot(W, A) + self.bias[i], dtype='float64')
            A = self.act_funcs[i].activation_func(Z)
            self.Zs.append(Z)
            self.As.append(A)
        return A
    
    
    def _backwardPropagation(self, predY, groundY):
        dAL = - (np.divide(groundY, predY) - np.divide(1 - groundY, 1 - predY))
        for i in reversed(range(self._layer_no)):
            ZL = self.Zs[i]
            dZL = np.multiply(dAL, self.act_funcs[i].activation_func_drv(ZL))
            AL1 = self.As[i] # A[l - 1]
            m = len(predY)
            dWL = (1 / m) * np.dot(dZL, AL1.T)
            dbL = (1 / m) * np.sum(dZL, axis=1, keepdims=True)
            self.weights[i] = self.weights[i] - self.learning_rate * dWL
            self.bias[i] = self.bias[i] - self.learning_rate * dbL
            dAL = np.dot(self.weights[i].T, dZL)
        
        
    def compileModel(self, optimizer='gradient_descent', loss='cross_entropy', 
                           epoch=10, learning_rate=0.0001):
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = epoch
        self.learning_rate = learning_rate


    def _calculateCostFunction(self, predY, groundY):
        if self.loss == 'cross_entropy':
            m = len(predY)
            error = np.squeeze((- 1 / m) * np.sum(np.multiply(groundY, np.log(predY)) + 
                                      np.multiply((1 - groundY), np.log(1 - predY))))
            print('error: ', error)
            self.errors.append(error) 
            return error
        else:   
            print('Wrong typed cost function!')
            return

    
    def _softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z))
    
        
    def _leaky_relu(self, Z):
        return np.maximum(Z, 0.01*Z)        

