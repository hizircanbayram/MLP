import numpy as np
import matplotlib.pyplot as plt

# plotting cosmetics
#%config InlineBackend.figure_format = 'svg' 
#plt.style.use('bmh')

# example of a linear model
from sklearn.linear_model import Perceptron

# example of a tree model
from sklearn.tree import DecisionTreeClassifier


np.random.seed(0)

def make_data(n):
    Y = np.random.choice([-1, +1], size=n)
    X = np.random.normal(size = (n, 2))
    for i in range(len(Y)):
        X[i] += Y[i]*np.array([-2, 0.9])
    return X, Y

def plot_decision_boundary(mlr, X, Y, cmap='Paired_r'):
    h = 2
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
    Z = np.array(Z).reshape((len(Z), 1))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    par1 = X[:,0]
    par2 = X[:,1]
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k');


X, Y = make_data(100)

plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=Y, cmap='Paired_r', edgecolors='k');

perc = Perceptron(max_iter=5)
perc.fit(X, Y)

tree = DecisionTreeClassifier()
tree.fit(X, Y);

perc.predict([[3, 2]]), tree.predict([[3, 2]])


    
    
plot_decision_boundary(perc, X, Y)