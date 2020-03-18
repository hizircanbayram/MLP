import sys
sys.path.append('../')
sys.path.append('../activations')
sys.path.append('../optimizers')

from neural_network import NeuralNetwork

from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np

from activations.relu import relu
from activations.sigmoid import sigmoid
from activations.tanh import tanh

from optimizers.gradient_descent import gd



datas = pd.read_csv('iris_nn.data', sep=',',header=None).to_numpy()
np.random.shuffle(datas)
X = datas[:, 0:4].T
X = normalize(X)
Y = datas[:, 4:5]

for i, y in enumerate(Y):
    if y == 'Iris-setosa':
        Y[i] = 0
    else:
        Y[i] = 1

Y = Y.T

model = NeuralNetwork()  
model.createLayer(8, input_dim=4, act_func=relu())
model.createLayer(8, act_func=tanh())
model.createLayer(8, act_func=relu())
model.createLayer(1, act_func=sigmoid())
model.compileModel(optimizer=gd(), loss='cross_entropy', 
                   epoch=1000)
model.train(X,Y)
a_last = model._forwardPropagation(X)
    