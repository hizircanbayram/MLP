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
from activations.softmax import softmax

from initializations.he_init import he_init
from initializations.xavier_init import xavier_init

from cost_functions.categorical_crossentropy import categorical_crossentropy

from optimizers.Adam import Adam

from measuring_metrics import *
from utils import *

datas = pd.read_csv('iris_nn.data', sep=',',header=None).to_numpy()
np.random.shuffle(datas)
X = datas[:, 0:4]
X = normalize(X)
Y = datas[:, 4:5]

for i, y in enumerate(Y):
    if y == 'Iris-setosa':
        Y[i] = 0
    elif y == 'Iris-versicolor':
        Y[i] = 1
    else:
        Y[i] = 2
        
Y = np.array(Y, dtype='int32')
Y_onehot = convert_one_hot(Y)

X_train = X[0:120,:]
X_test = X[120:150,:]

Y_train = Y_onehot[0:120,:]
Y_test = Y_onehot[120:150,:]

model = NeuralNetwork()  
model.createLayer(8, input_dim=4, act_func=relu(), weight_init=he_init())
model.createLayer(8, act_func=tanh(), weight_init=xavier_init())
model.createLayer(8, act_func=relu(), weight_init=he_init())
model.createLayer(3, act_func=softmax())
model.compileModel(optimizer=Adam(), loss_func=categorical_crossentropy(), 
                   epoch=5000)
model.train(X_train,Y_train)
Y_pred = model.predict(X_test)
print(measureAccuracy(Y_pred, Y_test))
plotLearningGraph(model.getErrors())
