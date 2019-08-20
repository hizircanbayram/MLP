from multi_variate_logistic_regression import MultivariateLogisticRegression 
from multi_variate_linear_regression import MultivariateLinearRegression
from measuring_metrics import MeasurementMetrics

import pandas as pd
        
datas = pd.read_csv('getting started toy datasets/multivariate_logistic_regression_data.txt').to_numpy()
X = datas[:, 0:2]
Y = datas[:, 2:3]


mm = MeasurementMetrics()
X_train, X_test, Y_train, Y_test = mm.seperateTrainTest(X, Y)

mlr = MultivariateLinearRegression(learning_rate=0.0000001, 
                                   learning_algorithm='Gradient Descent',
                                   epoch_num=200000, 
                                   reg_rate=None)
mlr.train(X_train, Y_train)
mm.stats(mlr, X_train, Y_train, X_test, Y_test, poly_deg=None)
mm.plotDecisionBoundary(mlr, X, Y)