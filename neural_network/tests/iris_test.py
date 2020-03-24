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

from measuring_metrics import *

datas = pd.read_csv('iris_nn.data', sep=',',header=None).to_numpy()
np.random.shuffle(datas)
X = datas[:, 0:4]
X = normalize(X)
Y = datas[:, 4:5]

for i, y in enumerate(Y):
    if y == 'Iris-setosa':
        Y[i] = 0
    else:
        Y[i] = 1

X_train = X[0:70,:]
X_test = X[70:100,:]

Y_train = Y[0:70,:]
Y_test = Y[70:100,:]

model = NeuralNetwork()  
model.createLayer(8, input_dim=4, act_func=relu())
model.createLayer(8, act_func=tanh())
model.createLayer(8, act_func=relu())
model.createLayer(1, act_func=sigmoid())
model.compileModel(optimizer=gd(), loss='cross_entropy', 
                   epoch=50)
model.train(X_train,Y_train)
Y_pred = model.predict(X_test)
print(measureAccuracy(Y_pred, Y_test))
plotLearningGraph(model.getErrors())
