import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt

#%% Reading in the Data
x_train = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/hw1-data/X_train.csv', header = None)
y_train = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/hw1-data/y_train.csv', header = None)
x_test = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/hw1-data/X_test.csv', header = None)
y_test = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/hw1-data/y_test.csv', header = None)
#%% Ridge regression function
def ridgereg(x, y, lmb):
    x2 = np.matmul(np.transpose(x), x)
    xy = np.matmul(np.transpose(x), y)
    chunk = lmb*np.identity(len(x2)) + x2
    chunk_inv = linalg.inv(chunk)
    return np.matmul(chunk_inv, xy)

#%% Degrees of Freedom Function
def df_lmb(x, y, lmb):
    x2 = np.matmul(np.transpose(x), x)
    chunk = lmb*np.identity(len(x2)) + x2
    chunk_inv = linalg.inv(chunk)
    prod_0 = np.matmul(x, chunk_inv)
    prod_1 = np.matmul(prod_0, np.transpose(x))
    return np.matrix.trace(prod_1)

#%% Parameters
params = np.zeros(shape = (5001, 7))
for i in range(5001):
    params[i] = np.transpose(ridgereg(x_train, y_train, i))

#%% Degrees of Freedom
df = np.zeros(shape = (5001, 1))
for i in range(5001):
    df[i] = df_lmb(x_train, y_train, i)
 
#%% Plotting w_rr as a function of df(lambda)
for col in range(7):
    plt.plot(df, params[:,col], label = col+1)
plt.ylabel('W_rr')
plt.xlabel('Degrees of Freedom')
plt.legend()
plt.show()
#%% Predictions and RMSE function
pred0 = np.matmul(x_test, np.transpose(params[0:51]))

def rmse(y, pred):
    diff = y - pred
    MSE = 1/42 * np.dot(np.transpose(diff), diff)
    return np.sqrt(MSE)

#%% Generate list of RMSE values for each lambda value from 0 to 50
rmse_list = np.zeros(shape = (51, 1))
for i in range(51):
    pred_current = pred0[:,i].reshape(42, 1)
    rmse_list[i] = rmse(y_test, pred_current)

#%% Plot graph of RMSE against lambda
lmb = list(range(51))
plt.plot(lmb, rmse_list) 
plt.ylabel('RMSE')
plt.xlabel('Lambda')
plt.show()

#%% Create data matrices for p = 2

poly2 = x_train.copy()
for i in range(7, 13):
    poly2[i] = poly2[i-7]**2

poly2_test = x_test.copy()
for i in range(7, 13):
    poly2_test[i] = poly2_test[i-7]**2

#%% Create data matrices for p = 3
poly3 = poly2.copy()
for i in range(13, 19):
    poly3[i] = poly3[i-13]**3
    
poly3_test = poly2_test.copy()
for i in range(13, 19):
    poly3_test[i] = poly3_test[i-13]**3
    
#%% Training p = 2 model
    
params2 = np.zeros(shape = (501, 13))
for i in range(501):
    params2[i] = np.transpose(ridgereg(poly2, y_train, i))
    
#%% Training p = 3 model
    
params3 = np.zeros(shape = (501, 19))
for i in range(501):
    params3[i] = np.transpose(ridgereg(poly3, y_train, i))
    
#%% RMSE for p = 2 model
    
poly2_pred = np.matmul(poly2_test, np.transpose(params2))

rmse_list_2 = np.zeros(shape = (501, 1))
for i in range(501):
    pred_current = poly2_pred[:,i].reshape(42, 1)
    rmse_list_2[i] = rmse(y_test, pred_current)

#%% RMSE for p = 3 model
    
poly3_pred = np.matmul(poly3_test, np.transpose(params3))

rmse_list_3 = np.zeros(shape = (501, 1))
for i in range(501):
    pred_current = poly3_pred[:,i].reshape(42, 1)
    rmse_list_3[i] = rmse(y_test, pred_current)

#%% RMSE for p = 1 model (with lambda up to 500 this time)
    
pred_1 = np.matmul(x_test, np.transpose(params[0:501]))    
    
rmse_list_1 = np.zeros(shape = (501, 1))
for i in range(501):
    pred_current = pred_1[:,i].reshape(42, 1)
    rmse_list_1[i] = rmse(y_test, pred_current)

#%% Graph the RMSE of all 3 models against lambda
    
lmb1 = list(range(501))
plt.plot(lmb1, rmse_list_1, label = "p = 1")
plt.plot(lmb1, rmse_list_2, label = "p = 2") 
plt.plot(lmb1, rmse_list_3, label = "p = 3")
plt.ylabel('RMSE')
plt.xlabel('Lambda')
plt.legend()
plt.show()

#%%