# Development of measuring metric tools implemented by Hizir Can Bayram.

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd  # for preprocessing the data
import matplotlib.pyplot as plt  # for plotting the graph



class MeasurementMetrics():
      
    
    
    # Plots a graph showing the changes in cost function based on number of iterations through the training. Its x-axis is limited with the number given as parameter.
    # Param | x_axis : Length of x-axis of the graph
    def plotCostFunction(self, model, x_axis):
        iters = [None] * model.getEpochNum()
        for i in range(model.getEpochNum()):
            iters[i] = i
        plt.xlim([0, x_axis])
        plt.plot(iters, model.getCostVals())
        plt.show()
        
        
    # Gives the value of final cost function based on squared error cost function.
    # Return | : value of cost function created with latest parameters
    def getFinalCost(self, model):
        return model.getFinalCostValue()



    def calculateDifference(self, Y_pred, Y_test):
        difference = np.subtract(Y_pred, Y_test)
        absolute = abs(difference)
        return np.sum(absolute) / len(Y_test)
    
    
    
    def seperateTrainTest(self, X, Y, poly_deg=None):
        if poly_deg == None:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
        else:
            polynomial_features= PolynomialFeatures(degree=poly_deg)
            X_poly = polynomial_features.fit_transform(X)
            X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.25, random_state=0)

        return X_poly, Y, X_train, X_test, Y_train, Y_test
        
    
    
    def plotDecisionBoundary(self, model, X, Y, cmap='Paired_r'):
        if model.getName() == 'MultivariateLogisticRegression':
            if X.shape[1] == 2:
                h = 2
                x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
                y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))
                par1 = xx.ravel()
                par2 = yy.ravel()
                par = np.c_[par1, par2]
                Z = []
                i = 0
                for x in par:
                    Z.append(model.predict([x]))
                    i += 1
                Z = np.array(Z)
                Z = Z.reshape(xx.shape)
                plt.figure(figsize=(5,5))
                plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
                plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
                par1 = X[:,0]
                par2 = X[:,1]
                Y = Y.reshape((len(Y)))
                plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k');
            else:
                print('Decision boundary can\'t be drawn for more than 2 features')
        else:
            print('Decision Boundry can only be applied to classification models')
        plt.show()  
        
        
    def stats(self, msg, model, X_train, Y_train, X_test, Y_test, poly_deg=None):    
        Y_pred_test = []    
        for sample in X_test:
            Y_pred_test.append(model.predict([sample]))
        
        Y_pred_test = np.asarray(Y_pred_test).reshape((len(Y_pred_test), 1))
        #rmse_test = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
        #r2_test = r2_score(Y_test, Y_pred_test)
        
        Y_pred_train = []    
        for sample in X_train:
            Y_pred_train.append(model.predict([sample]))
        
        Y_pred_train = np.asarray(Y_pred_train).reshape((len(Y_pred_train), 1))
        #rmse_train = np.sqrt(mean_squared_error(Y_train, Y_pred_train))
        #r2_train = r2_score(Y_train, Y_pred_train)    
        
        print(msg)
        print('learn_rate : ', model.getLearningRate())
        print('reg_rate : ', model.getRegRate())
        print('epoch : ', model.getEpochNum())
        print('poly_deg : ', poly_deg)   
        
        print('----- ----- TEST ----- -----')
        print('error amount for a single sample : {0:.2f}'.format(self.calculateDifference(Y_pred_test, Y_test)))
        print('final loss :', "{0:.2f}".format(self.getFinalCost(model), 2))
        #print('rmse : ', rmse_test)
        #print('r2 : ', r2_test)
    
        print('----- ----- TRAIN ----- -----')
        print('error amount for a single sample : {0:.2f}'.format(self.calculateDifference(Y_pred_train, Y_train), 2))
        print("final loss : {0:.2f}".format(self.getFinalCost(model), 2))
        #print('rmse : ', rmse_train)
        #print('r2 : ', r2_train)
        
        if model.getName() == 'MultivariateLogisticRegression':
            true = 0
            for i in range(len(X_test)):
                if Y_test[i] == Y_pred_test[i]:
                    true += 1
            print('True prediction is : ', true, '/', len(X_test), ' for TEST')
            
            true = 0
            for i in range(len(X_train)):
                if Y_train[i] == Y_pred_train[i]:
                    true += 1
            print('True prediction is : ', true, '/', len(X_train), ' for TRAIN')
            
                    