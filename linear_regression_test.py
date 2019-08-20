from multi_variate_linear_regression import MultivariateLinearRegression 

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd  # for preprocessing the data
import matplotlib.pyplot as plt  # for plotting the graph
import seaborn as sns 


def plot_decision_boundary(mlr, X, Y, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    par1 = xx.ravel()
    par2 = yy.ravel()
    par = np.c_[par1, par2]
    Z = []
    for x in par:
        Z.append(mlr.predict([x]))
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    par1 = X[:,0]
    par2 = X[:,1]
    Y = Y.reshape((len(Y)))
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k');
    
    
def stats(X, Y, reg_rate, learning_rate, epoch, poly_deg, msg):
    if poly_deg == None:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    else:
        polynomial_features= PolynomialFeatures(degree=poly_deg)
        X_poly = polynomial_features.fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.25, random_state=0)
            
    mlr = MultivariateLinearRegression(learning_rate, 'Gradient Descent', epoch, reg_rate)
    mlr.train(X_train, Y_train)

    Y_pred_test = []    
    for sample in X_test:
        Y_pred_test.append(mlr.predict([sample]))
    
    Y_pred_test = np.asarray(Y_pred_test).reshape((len(Y_pred_test), 1))
    rmse_test = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
    r2_test = r2_score(Y_test, Y_pred_test)
    
    Y_pred_train = []    
    for sample in X_train:
        Y_pred_train.append(mlr.predict([sample]))
    
    Y_pred_train = np.asarray(Y_pred_train).reshape((len(Y_pred_train), 1))
    rmse_train = np.sqrt(mean_squared_error(Y_train, Y_pred_train))
    r2_train = r2_score(Y_train, Y_pred_train)    
    
    print(msg)
    print('learn_rate : ', learning_rate, end=' ')
    print('reg_rate : ', reg_rate, end=' ')
    print('epoch : ', epoch, end=' ')
    print('poly_deg : ', poly_deg)   
    
    print('----- ----- TEST ----- -----')
    print('cost : ', mlr.calculateDifference(Y_pred_test, Y_test))
    print('final cost : ', mlr.getFinalCost())
    print('rmse : ', rmse_test)
    print('r2 : ', r2_test)

    print('----- ----- TRAIN ----- -----')
    print('cost : ', mlr.calculateDifference(Y_pred_train, Y_train))
    print('final cost : ', mlr.getFinalCost())
    print('rmse : ', rmse_train)
    print('r2 : ', r2_train)

    plot_decision_boundary(mlr, X_test, Y_test)


datas = pd.read_csv('getting started toy datasets/boston_housing.csv')
#sns.heatmap(data=datas.corr().round(1), annot=True)
rm = datas['rm'].to_numpy()
rm = rm.reshape((len(rm), 1))
lstat = datas['lstat'].to_numpy()
lstat = lstat.reshape((len(lstat), 1))
linear_X = np.concatenate((rm, lstat), 1)
Y = datas['medv'].to_numpy()
Y = Y.reshape((len(Y), 1))



stats(linear_X, Y, reg_rate=80, 
      learning_rate=0.000001, epoch=200, poly_deg=None,
      msg='POLY')