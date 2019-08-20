import sys
sys.path.append('../')

from multi_variate_linear_regression import MultivariateLinearRegression
from measuring_metrics import MeasurementMetrics

import pandas as pd
import numpy as np

datas = pd.read_csv('../getting started toy datasets/boston_housing.data')
X = pd.DataFrame(np.c_[datas['lstat'], datas['rm']], columns = ['lstat','rm']).to_numpy()
Y = datas['medv'].to_numpy()

mm = MeasurementMetrics()
X, Y, X_train, X_test, Y_train, Y_test = mm.seperateTrainTest(X, Y, poly_deg=None)

mlr = MultivariateLinearRegression(learning_rate=0.000001,
                                   learning_algorithm='Gradient Descent',
                                   epoch_num=1000000,
                                   reg_rate=0.09)
mlr.train(X_train, Y_train)

mm.stats('----- ----- BOSTON HOUSING ----- -----', mlr, X_train, Y_train, X_test, Y_test, poly_deg=None)
mm.plotDecisionBoundary(mlr, X, Y)
mm.plotCostFunction(mlr, mlr.getEpochNum())


Y_pred = []
for x in X_test:
    Y_pred.append(mlr.predict([x]))