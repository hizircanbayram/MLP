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

from optimizers.Adam import Adam

from cost_functions.binary_crossentropy import binary_crossentropy

from initializations.he_init import he_init

from measuring_metrics import *

datas = pd.read_csv('university_grades.data', sep=',',header=None).to_numpy()
np.random.shuffle(datas)
X = datas[:, 0:2]
X = normalize(X)
Y = datas[:, 2:3]

X_train = X[0:90, :]
X_test = X[90:100, :]

Y_train = Y[0:90, :]
Y_test = Y[90:100, :]

model = NeuralNetwork()  
model.createLayer(8, input_dim=2, act_func=relu(), weight_init=he_init())
model.createLayer(8, act_func=relu(), weight_init=he_init())
model.createLayer(1, act_func=sigmoid())
model.compileModel(optimizer=Adam(), loss_func=binary_crossentropy(), 
                   epoch=20000)
model.train(X_train, Y_train)
Y_pred = model.predict(X_test)
print(measureAccuracy(Y_pred, Y_test))
plotLearningGraph(model.getErrors())