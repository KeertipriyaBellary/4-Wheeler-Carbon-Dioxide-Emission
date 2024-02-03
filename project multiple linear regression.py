import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df=pd.read_csv(r"F:\M.tech\2nd sem\data\carfuel.csv")

cdf = df[['Engine size','Cylinders','Combined','CO2 emissions', 'City', 'Highway', 'Smog rating']]
cdf=cdf.rename(columns={"Engine size":"ENGINESIZE","Cylinders":"CYLINDERS","CO2 emissions":"CO2EMISSIONS"})

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
from sklearn import linear_model
regr = linear_model.LinearRegression()
#model1

train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS', 'Combined', 'Smog rating']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
from sklearn.metrics import r2_score
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','Combined', 'Smog rating']])
test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','Combined', 'Smog rating']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
print("Mean Squared Error (MSE) : %.2f"
      % np.mean((y_hat - test_y) ** 2))
print('Variance score: %.2f' % regr.score(test_x, test_y))

#model2

train_x2 = np.asanyarray(train[['ENGINESIZE','CYLINDERS', 'City', 'Highway', 'Smog rating']])
train_y2 = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x2, train_y2)

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
y_hat2= regr.predict(test[['ENGINESIZE','CYLINDERS', 'City', 'Highway', 'Smog rating']])
test_x2 = np.asanyarray(test[['ENGINESIZE','CYLINDERS', 'City', 'Highway', 'Smog rating']])
test_y2 = np.asanyarray(test[['CO2EMISSIONS']])
print("Mean Squared Error (MSE) (model 2)  : %.2f"
      % np.mean((y_hat2 - test_y2) ** 2))
print('Variance score (model 2): %.2f' % regr.score(test_x2, test_y2))

#model3

train_x3 = np.asanyarray(train[['ENGINESIZE','CYLINDERS', 'Combined']])
train_y3 = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x3, train_y3)

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
from sklearn.metrics import r2_score
y_hat3= regr.predict(test[['ENGINESIZE','CYLINDERS','Combined']])
test_x3 = np.asanyarray(test[['ENGINESIZE','CYLINDERS','Combined']])
test_y3= np.asanyarray(test[['CO2EMISSIONS']])
print("Mean Squared Error (MSE) (model 3): %.2f"
      % np.mean((y_hat3 - test_y3) ** 2))
print('Variance score (model 3): %.2f' % regr.score(test_x3, test_y3))
