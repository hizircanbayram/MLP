
from multi_variate_logistic_regression import MultivariateLogisticRegression 

import pandas as pd  # for preprocessing the data
import matplotlib.pyplot as plt  # for plotting the graph

datas = pd.read_csv('getting started toy datasets/multivariate_logistic_regression_data.txt').to_numpy()
X = datas[0:75,0:2]
Y = datas[0:75:,2:3]
X_test = datas[75:100, 0:2]
Y_test = datas[75:100, 2:3]

mlr = MultivariateLogisticRegression(0.0001, 1000000) 
tets = mlr.train(X, Y)


Y_pred = []
for x in X_test:
    Y_pred.append(mlr.predict([x]))
    
true = 0
for i in range(len(Y_pred)):
    if Y_pred[i] == Y_test[i]:
        true += 1
        
print('true : ', true)

#mlr.plotCostFunction(1000000)
