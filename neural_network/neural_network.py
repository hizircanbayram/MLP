import numpy as np
import sys
sys.path.append('optimizers/') # might be needed
sys.path.append('initializers/')
sys.path.append('activations/')

from optimizers.gradient_descent import gd
from activations.relu import relu
from initializations.standart_normal import standart_normal

class NeuralNetwork():

    # DO ANOTHER MODULE FOR INITIALIZATION(it causes the NN ends up nan or not dramatically, solve this)
    # ADD ANOTHER ACTIVATIONS SUCH AS LEAKY_RELU, SOFTMAX
    # DO ANOTHER MODULE FOR LOSS FUNCTION
    # SAVING & LOADING MODELS
    # REAL TIME HAND DIGIT CLASSIFIER USING THIS FRAMEWORK
    
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
        
        
    def createLayer(self, neuron_size, act_func=relu(), weight_init=standart_normal(), input_dim=None):
        if self._layer_no == 0 and input_dim == None:
            print('Input size has to be determined')
            return
        if self._layer_no != 0 and input_dim != None:
            print('Input size must not be determined')
            return
        # Either granted, self._init_cnt != 0 or input_dim != None. Not both
        if input_dim != None: # First layer
            created_weight = weight_init.initializeWeights(neuron_size, input_dim)
        if self._layer_no != 0:
            second_dim = self.weights[self._layer_no - 1].shape[0]
            created_weight = weight_init.initializeWeights(neuron_size, second_dim)
        self.weights.append(created_weight)
        self.bias.append(np.zeros((neuron_size, 1)))
        self.act_funcs.append(act_func)
        self._layer_no += 1
        
        
    def train(self, X, Y):
        '''
        forward and backward prop. assume the samples in the dataset are set in
        the vertical way because of the easier vectorization. However, it is easier 
        for people to think a dataset in the horizontal way since it is inituitive 
        to access the first element of a dataset by dataset[0] instead of a[:,0].
        So X and Y are passed in horizantal way and they then converte to vertical way.
        '''
        X = X.T
        Y = Y.T
        for i in range(self.epoch):
            predY = self._forwardPropagation(X)
            error = self._calculateCostFunction(predY, Y)
            self._logHelper(i, error, self.predict(X.T), Y.T)
            self._backwardPropagation(predY, Y)            
            self.As = []
            self.Zs = []


    def predict(self, X):
        A = X.T # check the note in train method, same explanation.
        for i in range(self._layer_no):
            W = self.weights[i]
            Z = np.asarray(np.dot(W, A) + self.bias[i], dtype='float64')
            A = self.act_funcs[i].activation_func(Z)
        return np.around(A).T # check the note in train method, same explanation.
    
    
    def compileModel(self, optimizer=gd(), loss='cross_entropy', epoch=10):
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = epoch
        

    def getErrors(self):
        return self.errors
    
    
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
        new_weights, new_bias = self.optimizer._backwardPropagation(predY, groundY, self)
        self.weights = new_weights
        self.bias = new_bias
      
        
    def _log_epoch_helper(self, boundry, epoch_no, error, Y_pred, Y_train):
        # if epoch_no % (boundry // 10) == 0, this means it'll log the current
        # situation ten times for the entire training
        if (epoch_no == self.epoch - 1) or (epoch_no % (boundry // 10) == 0):
            print(str(epoch_no + 1) + '.', 'epoch, error:', 
                  "{0:,.3f}".format(error), 'train acc: ',
                  "{0:,.3f}".format(100 - (100 * np.sum(np.absolute(Y_pred - Y_train))) / len(Y_train)))    
    
        
    def _logHelper(self, current_epoch, error, Y_pred, Y_train):
        '''
        based on total epoch number, this function determines in which epoch
        the infos will be printed
        '''
        if self.epoch <= 10:
            self._log_epoch_helper(10, current_epoch, error, Y_pred, Y_train)
        elif self.epoch <= 100:
            self._log_epoch_helper(100, current_epoch, error, Y_pred, Y_train)
        elif self.epoch <= 500:
            self._log_epoch_helper(500, current_epoch, error, Y_pred, Y_train)
        elif self.epoch <= 1000:
            self._log_epoch_helper(1000, current_epoch, error, Y_pred, Y_train)
        elif self.epoch <= 5000:
            self._log_epoch_helper(5000, current_epoch, error, Y_pred, Y_train)
        elif self.epoch <= 10000:
            self._log_epoch_helper(10000, current_epoch, error, Y_pred, Y_train)
        elif self.epoch <= 100000:
            self._log_epoch_helper(100000, current_epoch, error, Y_pred, Y_train)
        else:
            self._log_epoch_helper(100000, current_epoch, error, Y_pred, Y_train)
        

    def _calculateCostFunction(self, predY, groundY):
        if self.loss == 'cross_entropy':
            m = len(predY)
            error = np.squeeze((- 1 / m) * np.sum(np.multiply(groundY, np.log(predY)) + 
                                      np.multiply((1 - groundY), np.log(1 - predY))))
            self.errors.append(error) 
            return error
        else:   
            print('Wrong typed cost function!')
            return

    
    def _softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z))
    
        
    def _leaky_relu(self, Z):
        return np.maximum(Z, 0.01*Z)        

