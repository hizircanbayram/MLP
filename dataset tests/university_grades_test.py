import sys
sys.path.append('../')

from multi_variate_logistic_regression import MultivariateLogisticRegression 
from measuring_metrics import MeasurementMetrics

import pandas as pd

datas = pd.read_csv('../getting started toy datasets/university_grades.data').to_numpy()
X = datas[:, 0:2]
Y = datas[:, 2:3]

mm = MeasurementMetrics()
X, Y, X_train, X_test, Y_train, Y_test = mm.seperateTrainTest(X, Y, poly_deg=None)

mlr = MultivariateLogisticRegression(learning_rate=0.001,
                                     epoch_num=200000,
                                     reg_rate=None)
mlr.train(X_train, Y_train)

mm.stats('----- ----- UNIVERSITY GRADES ----- -----', mlr, X_train, Y_train, X_test, Y_test, poly_deg=None)
mm.plotDecisionBoundary(mlr, X, Y)
mm.plotCostFunction(mlr, mlr.getEpochNum())