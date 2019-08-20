import sys
sys.path.append('../')

from multi_variate_logistic_regression import MultivariateLogisticRegression 
from measuring_metrics import MeasurementMetrics

import pandas as pd

datas = pd.read_csv('../getting started toy datasets/iris.data').to_numpy()
X = datas[:, 0:4]
Y = datas[:, 4:5]

mm = MeasurementMetrics()
X, Y, X_train, X_test, Y_train, Y_test = mm.seperateTrainTest(X, Y, poly_deg=None)

mlr = MultivariateLogisticRegression(learning_rate=0.01,
                                     epoch_num=10000,
                                     reg_rate=0.09)
mlr.train(X_train, Y_train)

mm.stats('----- ----- IRIS ----- -----', mlr, X_train, Y_train, X_test, Y_test, poly_deg=None)
mm.plotDecisionBoundary(mlr, X, Y)
mm.plotCostFunction(mlr, mlr.getEpochNum())
