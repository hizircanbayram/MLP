from multi_variate_linear_regression import MultivariateLinearRegression 

import pandas as pd  # for preprocessing the data
import matplotlib.pyplot as plt  # for plotting the graph

datas = pd.read_csv('getting started toy datasets/multivariate_linear_regression_data.txt').to_numpy()
X = datas[0:25,0]
Y = datas[0:25:,1]
X_test = datas[25:29, 0]
Y_test = datas[25:29, 1]

mlr = MultivariateLinearRegression(0.0001, 'Gradient Descent', 1000000) 
mlr.train(X, Y)

Y_pred = []
for x in X_test:
    Y_pred.append(mlr.predict([x]))
    
plt.plot(X, Y, 'bs')
plt.plot(X_test, Y_pred, 'r') 

