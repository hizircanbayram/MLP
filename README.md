# MLP

## MLP is a machine learning library implemented in Python. 

### Machine Learning algorithms MLP covers are as follows:
  * Multivarite Linear Regression (done)
  * Logistic Regression with multiclass classification (to do)
  * Neural Networks (to do)
  * Support Vector Machines (to do)


### MLP also covers some classes to measure various metrics so that users can see their model's progress. These are as follows:
  * Sqaured error function to understand which parameters to the function are the best based on various model tries (to do)
  * Data visualization for every dimension in the dataset and dependent variable (to do)
  * Cost function - number of iterations graph (to do)
  * Confusion Matrix (to do)
  

### Getting started with MLP

#### Multivariate Linear Regression

Importing necessary modules

```python
from multi_variate_linear_regression import MultivariateLinearRegression 

import pandas as pd  # for preprocessing the data
import matplotlib.pyplot as plt  # for plotting the graph
```

Loading data, preparing it for the model using numpy

```python
datas = pd.read_csv('getting started toy datasets/multivariate_linear_regression_data.txt').to_numpy()
X = datas[0:25,0]
Y = datas[0:25:,1]
X_test = datas[25:29, 0]
Y_test = datas[25:29, 1]
```

Training model with variables

```python
mlr = MultivariateLinearRegression(0.0001, 'Gradient Descent', 1000000) 
mlr.train(X, Y)
```

Predicting different independent variables

```python
Y_pred = []
for x in X_test:
    Y_pred.append(mlr.predict([x]))
```

Plotting the dataset and model's graph

```python
plt.plot(X, Y, 'bs')
plt.plot(X_test, Y_pred, 'r') 
```

![image](https://user-images.githubusercontent.com/23126077/62820758-a70f3100-bb71-11e9-8479-b878c4247d4e.png)
    
### Prerequisites

* numpy


### Built With

* [Python](https://www.python.org/) 
