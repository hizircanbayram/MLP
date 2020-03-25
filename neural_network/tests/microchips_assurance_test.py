import sys
sys.path.append('../')
sys.path.append('../activations')
sys.path.append('../optimizers')

from neural_network import NeuralNetwork

import pandas as pd
import numpy as np

from activations.relu import relu
from activations.sigmoid import sigmoid
from activations.tanh import tanh

from optimizers.gradient_descent import gd

from cost_functions.binary_crossentropy import binary_crossentropy

from measuring_metrics import *

datas = pd.read_csv('microchips_assurance.data', sep=',',header=None).to_numpy()
np.random.shuffle(datas)
X = datas[:, 0:2]
Y = datas[:, 2:3]

X_train = X[0:95, :]
X_test = X[95:119, :]

Y_train = Y[0:95, :]
Y_test = Y[95:119, :]

model = NeuralNetwork()  
model.createLayer(8, input_dim=2, act_func=relu())
model.createLayer(16, act_func=tanh())
model.createLayer(16, act_func=relu())
model.createLayer(8, act_func=relu())
model.createLayer(1, act_func=sigmoid())
model.compileModel(optimizer=gd(), loss_func=binary_crossentropy(), 
                   epoch=10000)
model.train(X_train, Y_train)
Y_pred = model.predict(X_test)
print(measureAccuracy(Y_pred, Y_test))
plotLearningGraph(model.getErrors())