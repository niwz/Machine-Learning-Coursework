#%% Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%% Load in Data

X_train = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW3/hw3-data/gaussian_process/X_train.csv', header = None)
X_test = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW3/hw3-data/gaussian_process/X_test.csv', header = None)
y_train = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW3/hw3-data/gaussian_process/y_train.csv', header = None)
y_test = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW3/hw3-data/gaussian_process/y_test.csv', header = None)


#%%

X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

#%% a) Write code to implement the Gaussian process and to make predictions on test data.

class GaussianProcess(object):
    
    def kernel(self, x1, x2, b):
        return np.exp(-(np.linalg.norm(x1 - x2, 2)**2) / b)
    
    def fit(self, X, y, b):
        self.X = X.copy()
        self.y = y.copy()
        self.b = b
        self.Kn = np.zeros(shape = (len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                self.Kn[i][j] = self.kernel(X[i], X[j], b)
        
    def predict(self, X, sigma):
        K = np.zeros(shape = (len(X), len(self.X)))
        for i in range(len(X)):
            for j in range(len(self.X)):
                K[i][j] = self.kernel(X[i], self.X[j], self.b)
        self.mu = np.dot(np.dot(K, np.linalg.inv(np.add(sigma * np.eye(len(self.X)), self.Kn))), self.y)
    
    def score(self, y):
        diff = self.mu - y
        MSE = 1/len(y) * np.dot(np.transpose(diff), diff)
        return np.sqrt(MSE)
#%%

gp = GaussianProcess()
b_list = [5,7,9,11,13,15]
sigma2_list = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
#b_list = [5,7]
#sigma2_list = [.1,.2]
rmse_table = np.zeros(shape = (len(b_list), len(sigma2_list)))
for b in b_list:
    for s in sigma2_list:
        gp.fit(X_train, y_train, b)
        gp.predict(X_test, s)
        rmse_table[b_list.index(b)][sigma2_list.index(s)] = gp.score(y_test)
        
rmse_table

#%%

np.argmin(rmse_table) # Lowest RMSE achieved at index 30 ie b = 9, sigma^2 = 0.1

#%%

X4 = X_train[:, 3].copy().flatten()
X4 = X4.reshape(350, 1)
gp.fit(X4, y_train, 5)
gp.predict(X4, 2)

#%%
X4 = X4.flatten()
pred_mean = gp.mu.flatten()

#%%

plt.figure()
plt.scatter(X4[np.argsort(X4)], y_train[np.argsort(X4)])
plt.plot(X4[np.argsort(X4)], pred_mean[np.argsort(X4)],'r-')
plt.show()