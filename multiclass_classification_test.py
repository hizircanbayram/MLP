from multi_variate_logistic_regression import MultivariateLogisticRegression 

import pandas as pd  # for preprocessing the data
import matplotlib.pyplot as plt  # for plotting the graph
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('getting started toy datasets/iris_data.txt')
data = pd.DataFrame(dataset.iloc[:, 0:5].values, columns=['A', 'B', 'C', 'D', 'E'])

Y_vals = data['E']
X_vals = data[['A', 'B', 'C', 'D']]

X_train, X_test, Y_train, Y_test = train_test_split(X_vals, Y_vals, test_size=0.25, random_state=0)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

mlr = MultivariateLogisticRegression(0.0001, 1000000) 
aa = mlr.train(X_train, Y_train)


Y_pred = []
for x in X_test:
    Y_pred.append(mlr.predict([x]))
    
true = 0
for i in range(len(Y_pred)):
    if Y_pred[i] == Y_test[i]:
        true += 1
        
print('# of true test input is : ', true)