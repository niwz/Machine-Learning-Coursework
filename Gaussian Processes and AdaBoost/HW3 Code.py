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
    
#%% Generate results table for values given

gp = GaussianProcess()
b_list = [5,7,9,11,13,15]
sigma2_list = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
rmse_table = np.zeros(shape = (len(b_list), len(sigma2_list)))
for b in b_list:
    for s in sigma2_list:
        gp.fit(X_train, y_train, b)
        gp.predict(X_test, s)
        rmse_table[b_list.index(b)][sigma2_list.index(s)] = gp.score(y_test)
        
rmse_table
np.savetxt("Results Table.csv", rmse_table, delimiter=",")

#%% Find minimum RMSE

np.argmin(rmse_table) # Lowest RMSE achieved at index 30 ie b = 9, sigma^2 = 0.1

#%%

X4 = X_train[:, 3].copy().flatten()
X4 = X4.reshape(350, 1)
gp.fit(X4, y_train, 5)
gp.predict(X4, 2)

#%%
X4 = X4.flatten()
pred_mean = gp.mu.flatten()

#%% Plot the Gaussian Process
plt.figure()
plt.scatter(X4[np.argsort(X4)], y_train[np.argsort(X4)])
plt.plot(X4[np.argsort(X4)], pred_mean[np.argsort(X4)], c = 'red')
plt.xlabel('Car Weight')
plt.ylabel('Miles per Gallon')
plt.show()

#%% AdaBoost
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#%% Load in Data

X_train = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW3/hw3-data/boosting/X_train.csv', header = None)
X_test = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW3/hw3-data/boosting/X_test.csv', header = None)
y_train = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW3/hw3-data/boosting/y_train.csv', header = None)
y_test = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW3/hw3-data/boosting/y_test.csv', header = None)


#%% Convert to numpy array

X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

#%% Boosting

class AdaBoost(object):
    
    def __init__(self, X, y, X_test, y_test):
        self.X = np.column_stack([np.ones(len(X)), X])
        self.X_test = np.column_stack([np.ones(len(X_test)), X_test])
        self.y = np.where(y == 0, -1, y).flatten()
        self.y_test = np.where(y_test == 0, -1, y_test).flatten()
        self.weights = np.repeat(1/len(self.X), len(self.X)).flatten()
        self.upperbound_exp = 0
        self.total_train_pred = 0
        self.total_test_pred = 0
        self.times_sampled = np.zeros(self.X.shape[0])
        
        
    def fit(self):
        
        # bootstrap
        sample_index = np.random.choice(self.X.shape[0], self.X.shape[0], 
                                        replace=True, 
                                        p = list(np.asarray(self.weights).flatten()))
        self.X_sample = self.X[sample_index, :]
        self.y_sample = self.y[sample_index]
        self.w = np.dot(np.linalg.inv(np.dot(self.X_sample.T, self.X_sample)), 
                        np.dot(self.X_sample.T, self.y_sample))
        
        # how many times each data point was sampled from bootstrapping
        for i in sample_index:
            self.times_sampled[i] += 1
        
    def predict(self, X):
        return np.sign(np.dot(X, self.w))
                
        
    def update(self):
        
        # get training predictions
        predictions = self.predict(self.X)
        
        # calculate epsilon
        self.epsilon = np.sum(self.weights[~np.equal(self.y, predictions)])
        if self.epsilon > 0.5:
            self.epsilon = 1 - self.epsilon
            self.w = - self.w
            predictions = - predictions
        
        # calculate alpha                  
        self.alpha = 0.5 * np.log((1-self.epsilon) / self.epsilon)
        scaler = np.exp(- self.alpha * np.multiply(self.y, predictions))
        self.weights = np.multiply(self.weights, scaler)
        self.weights = self.weights / sum(self.weights)
        
        # calculate upper bound for training error
        self.upperbound_exp += (0.5-self.epsilon)**2
        self.upperbound = np.exp(-2 * self.upperbound_exp)
        
    def score(self):
        
        # running totals
        self.total_train_pred += self.alpha * self.predict(self.X)
        self.total_test_pred += self.alpha * self.predict(self.X_test)
        
        # errors
        train_error = np.mean(np.sign(self.total_train_pred)!= self.y)
        test_error = np.mean(np.sign(self.total_test_pred)!= self.y_test)
        
        return train_error, test_error
        
        
#%% Initialize the model
ab = AdaBoost(X_train, y_train, X_test, y_test)
ab.fit()

#%% Iterate 1500 times and get errors and upper bound
errors = np.zeros(shape = (1500, 2))
upperbound = np.zeros(1500)
epsilons = np.zeros(1500)
alphas = np.zeros(1500)

for i in range(1500):
    ab.update()
    errors[i] = ab.score()
    upperbound[i] = ab.upperbound
    epsilons[i] = ab.epsilon
    alphas[i] = ab.alpha
    ab.fit()
        
#%% Plot errors
t = np.arange(1, 1501)
plt.plot(t, errors[:, 0], label = 'Training Error')
plt.plot(t, errors[:, 1], c = 'r', label = 'Testing Error')
plt.xlabel('Iterations of Boosting')
plt.ylabel('Error')
plt.legend()
plt.show()

#%% Plot upper bound
plt.figure()
plt.plot(t, upperbound)
plt.xlabel('Iterations of Boosting')
plt.ylabel('Upper Bound on Training Error')
plt.show()

#%% Histogram of frequency
plt.figure()
plt.bar(range(1, ab.X.shape[0] + 1), ab.times_sampled)
plt.xlabel('Training Data Index')
plt.ylabel('Times Sampled')
plt.show()


#%% Plot epsilons
plt.figure()
plt.plot(t, epsilons)
plt.xlabel('Iterations of Boosting')
plt.ylabel('Epsilon')
plt.show()

#%% Plot alphas
plt.figure()
plt.plot(t, alphas)
plt.xlabel('Iterations of Boosting')
plt.ylabel('Alpha')
plt.show()
#%%