import sys
sys.path.append('../')

from multi_variate_logistic_regression import MultivariateLogisticRegression 
from measuring_metrics import MeasurementMetrics

import pandas as pd

datas = pd.read_csv('../getting started toy datasets/microchips_assurance.data').to_numpy()
X = datas[:, 0:2]
Y = datas[:, 2:3]

mm = MeasurementMetrics()
X, Y, X_train, X_test, Y_train, Y_test = mm.seperateTrainTest(X, Y, poly_deg=6)

mlr = MultivariateLogisticRegression(learning_rate=0.02,
                                     epoch_num=1000000,
                                     reg_rate=0.05)
mlr.train(X_train, Y_train)

mm.stats('----- ----- MICROCHIPS ASSURANCE ----- -----', mlr, X_train, Y_train, X_test, Y_test, poly_deg=6)
mm.plotCostFunction(mlr, mlr.getEpochNum())