import numpy as np

class NeuralNetwork():

    # DO ANOTHER MODULE FOR ACTIVATION FUNCTIONS
    # DO ANOTHER MODULE FOR OPTIMIZATION ALGORITHMS
    # MAKE SURE THE DATAS ARE NEXT TO EACH OTHER IN THE VERTICAL WAY
    # MAKE SURE IT IS OKAY TO HAVE BIAS WITH ZEROS BY CHECKING YOUR NOTES
    # MAKE SURE THE FORWARD PROPAGATION WORKS WITH DIFFERENT ACTIVATION FUNCTIONS
    # GET XIAVIAR WEIGHTS BY HANDS FROM TF AND APPLY AND SEE IF YOUR FORWARD PROP IS ACTUALLY WORKING
    class WeightParams():
        
        def __init__(self, weights, act_func):
            self.weights = weights
            self.act_func = act_func
            
        def getWeights(self):
            return self.weights
        
        def getActFunc(self):
            return self.act_func
            
    
    def __init__(self):
        self.weights = []
        self.bias = []
        self.init_cnt = 0 # keeps track of number of layers in the NN
        
        
    def createLayer(self, neuron_size, act_func='relu', input_dim=None):
        if self.init_cnt == 0 and input_dim == None:
            print('Input size has to be determined')
            return
        if self.init_cnt != 0 and input_dim != None:
            print('Input size must not be determined')
            return
        # Either granted, self._init_cnt != 0 or input_dim != None. Not both
        if input_dim != None: # First layer
            created_weight = self.WeightParams(np.random.rand(neuron_size, input_dim), act_func)
            self.weights.append(created_weight)
        if self.init_cnt != 0:
            second_dim = self.weights[self.init_cnt - 1].getWeights().shape[0]
            created_weight = self.WeightParams(np.random.rand(neuron_size, second_dim), act_func)
            self.weights.append(created_weight)
        self.bias.append(np.zeros((neuron_size, 1)))
        self.init_cnt += 1
        
        
    def train(self, X, Y, optimizer='gradient_descent'):
        output = self._forwardPropagation(X)
        print(output)
        
        
    def _calculateActivation(self, Z, act_func):
        if act_func == 'sigmoid':
            return self._sigmoid(Z)
        elif act_func == 'tanh':
            return self._tanh(Z)
        elif act_func == 'relu':
            return self._relu(Z)
        elif act_func == 'leaky_relu':
            return self._leaky_relu(Z)
        else:
            print('Wrong activation name. Written: ', act_func)
            return
        
        
    def _forwardPropagation(self, X):
        A = X
        for i in range(self.init_cnt):
            act_func = self.weights[i].getActFunc()
            W = self.weights[i].getWeights()
            Z = np.dot(W, A) + self.bias[i]
            A = self._calculateActivation(Z, act_func)
        return A
    
    
    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    
    def _tanh(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    
    
    def _relu(self, Z):
        return np.maximum(Z, 0)
    
    
    def _leaky_relu(self, Z):
        return np.maximum(Z, 0.01*Z)    
    
    
    def logWeights(self):
        for i in range(self.init_cnt):
            print(self.weights[i].getWeights())
            print()
        
    
def normalizeInput(X, axis):
    if axis != 0 and axis != 1:
        print('Wrong axis! Must be either 0 or 1')
        return
    mu = X.sum(axis=axis, keepdims=True) / X.shape[axis]
    X_squared = np.square(X)
    sigma2 = X_squared.sum(axis=axis, keepdims=True) / X_squared.shape[axis]
    return (X - mu) / np.sqrt(sigma2)
    


from sklearn.preprocessing import normalize
sampleX = np.array([[1,7,10,12], 
                    [6,5,6,100], 
                    [10,8,2,-1]])
sampleX = normalize(sampleX)

model = NeuralNetwork()  
model.createLayer(2, input_dim=3)
model.createLayer(3)
model.createLayer(1, 'sigmoid')
model.logWeights()
model.train(sampleX, None)
#model.logWeights()

