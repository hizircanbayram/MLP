import numpy as np
from activations.activation import activation
from activations.relu import relu
from activations.sigmoid import sigmoid
from activations.tanh import tanh

class NeuralNetwork():

    # DO ANOTHER MODULE FOR INITIALIZATION
    # DO ANOTHER MODULE FOR OPTIMIZATION ALGORITHMS
    
    class WeightParams():
        
        def __init__(self, weight, act_func):
            self.weight = weight
            self.act_func = act_func
            
        def getWeight(self):
            return self.weight
        
        def setWeight(self, weight):
            self.weight = weight
        
        def getActFunc(self):
            return self.act_func
            
    
    def __init__(self):
        self.weights = []
        self.bias = []
        self.init_cnt = 0 # keeps track of number of layers in the NN
        self.optimizer = None
        self.loss = None # loss(error) function's name
        self.epoch = 0
        self.errors = []
        self.Zs = []
        self.As = []
        
        
    def createLayer(self, neuron_size, act_func='relu', input_dim=None):
        if self.init_cnt == 0 and input_dim == None:
            print('Input size has to be determined')
            return
        if self.init_cnt != 0 and input_dim != None:
            print('Input size must not be determined')
            return
        # Either granted, self._init_cnt != 0 or input_dim != None. Not both
        if input_dim != None: # First layer
            created_weight = self.WeightParams(np.random.randn(neuron_size, input_dim), act_func)
            self.weights.append(created_weight)
            if neuron_size == 1:
                print(created_weight.getWeight())
        if self.init_cnt != 0:
            second_dim = self.weights[self.init_cnt - 1].getWeight().shape[0]
            created_weight = self.WeightParams(np.random.randn(neuron_size, second_dim), act_func)
            self.weights.append(created_weight)
            if neuron_size == 1:
                print(created_weight.getWeight())
        self.bias.append(np.zeros((neuron_size, 1)))
        self.init_cnt += 1
        
        
    def train(self, X, Y):
        for i in range(self.epoch):
            predY = self._forwardPropagation(X)
            error = self._calculateCostFunction(predY, Y)
            self._backwardPropagation(predY, Y)            
            self.As = []
            self.Zs = []

        
    def _forwardPropagation(self, X):
        A = X
        self.As.append(A)
        for i in range(self.init_cnt):
            act_func = self.weights[i].getActFunc()
            W = self.weights[i].getWeight()
            Z = np.asarray(np.dot(W, A) + self.bias[i], dtype='float64')
            A = self.weights[i].getActFunc().activation_func(Z)
            self.Zs.append(Z)
            self.As.append(A)
        return A
    
    
    def _backwardPropagation(self, predY, groundY):
        dAL = - (np.divide(groundY, predY) - np.divide(1 - groundY, 1 - predY))
        for i in reversed(range(self.init_cnt)):
            ZL = self.Zs[i]
            act_func = self.weights[i].getActFunc()
            dZL = np.multiply(dAL, self.weights[i].getActFunc().activation_func_drv(ZL))
            AL1 = self.As[i]
            m = len(predY)
            dWL = (1 / m) * np.dot(dZL, AL1.T)
            dbL = (1 / m) * np.sum(dZL, axis=1, keepdims=True)
            self.weights[i].setWeight(self.weights[i].getWeight() - self.learning_rate * dWL)
            self.bias[i] = self.bias[i] - self.learning_rate * dbL
            dAL = np.dot(self.weights[i].getWeight().T, dZL)
        
        
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
            self.errors.append(error) # IT SHOULD'VE BEEN INSIDE THE COST FUNCTION WITHOUT RETURNING ERROR!
            return error
        else:   
            print('Wrong typed cost function!')
            return

    
    def _softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z))
    
        
    def _leaky_relu(self, Z):
        return np.maximum(Z, 0.01*Z)    
    
    
    def logWeights(self):
        for i in range(self.init_cnt):
            print(self.weights[i].getWeight())
            print()
            
            
    def setWeights(self):
        self.weights[0].setWeight(np.array([[ 0.35006494,  0.99620589,  0.05740815],
                                            [-0.9069341,   0.27929668,  0.64960634]]))
        self.weights[1].setWeight(np.array([[ 0.35006494,  0.99620589],
                                            [ 0.05740815, -0.9069341 ],
                                            [ 0.27929668,  0.64960634]]))
        self.weights[2].setWeight(np.array([[0.39138451, 1.11379205, 0.06418427]]))
        
    
def normalizeInput(X, axis):
    if axis != 0 and axis != 1:
        print('Wrong axis! Must be either 0 or 1')
        return
    mu = X.sum(axis=axis, keepdims=True) / X.shape[axis]
    X_squared = np.square(X)
    sigma2 = X_squared.sum(axis=axis, keepdims=True) / X_squared.shape[axis]
    return (X - mu) / np.sqrt(sigma2)
    


from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.model_selection import train_test_split
'''
sampleX = np.array([[1,7,10,12], 
                    [6,5,6,100], 
                    [10,8,2,-1]])
sampleX = normalize(sampleX)
'''

datas = pd.read_csv('../getting started toy datasets/microchips_assurance.data', sep=',',header=None).to_numpy()
np.random.shuffle(datas)
X = datas[:, 0:2].T
#X = normalize(X)
Y = datas[:, 2:3]
'''
for i, y in enumerate(Y):
    if y == 'Iris-setosa':
        Y[i] = 0
    else:
        Y[i] = 1
'''
Y = Y.T
model = NeuralNetwork()  
model.createLayer(8, input_dim=2, act_func=relu())
model.createLayer(8, act_func=tanh())
model.createLayer(8, act_func=relu())
model.createLayer(1, act_func=sigmoid())
model.compileModel(optimizer='gradient_descent', loss='cross_entropy', 
                   epoch=10000, learning_rate=0.0001)
model.train(X,Y)
a_last = model._forwardPropagation(X)
#model.logWeights()


    