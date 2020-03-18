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


datas = pd.read_csv('university_grades.data', sep=',',header=None).to_numpy()
np.random.shuffle(datas)
X = datas[:, 0:2].T
X = normalize(X)
Y = datas[:, 2:3].T

model = NeuralNetwork()  
model.createLayer(8, input_dim=2, act_func=relu())
model.createLayer(8, act_func=relu())
model.createLayer(1, act_func=sigmoid())
model.compileModel(optimizer=gd(), loss='cross_entropy', 
                   epoch=10000)
model.train(X,Y)
a_last = model._forwardPropagation(X)
