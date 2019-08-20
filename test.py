
from multi_variate_logistic_regression import MultivariateLogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd  # for preprocessing the data
import matplotlib.pyplot as plt  # for plotting the graph
import numpy as np

def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k');

datas = pd.read_csv('getting started toy datasets/regularization_data.txt').to_numpy()
X = datas[:,0:2]
Y = datas[:,2:3]

poly_features = PolynomialFeatures(degree=8)
X_poly = poly_features.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.25, random_state=0)

mlr = MultivariateLogisticRegression(0.1, 40000, 0.5) 
mlr.train(X_train, Y_train)

Y_pred_test = []

for x in X_test:
    Y_pred_test.append(mlr.predict([x]))
    
    
    
true = 0

for i in range(len(X_test)):
    if Y_pred_test[i] == Y_test[i]:
        true += 1

        

print('True : ', true / len(Y_test))        
        
true = 0

Y_pred_train = []

for x in X_train:
    Y_pred_train.append(mlr.predict([x]))

for i in range(len(X_train)):
    if Y_pred_train[i] == Y_train[i]:
        true += 1
        
print('True : ', true / len(Y_train))


'''
rmse_test = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2_test = r2_score(Y_test, Y_pred)

print('Rmse : ', rmse_test)
print('R2 : ', r2_test)
'''