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
plt.plot(t, errors[:, 0])
plt.plot(t, errors[:, 1], c = 'r')
plt.show()

#%% Plot upper bound
plt.figure()
plt.plot(t, upperbound)
plt.show()

#%% Plot epsilons
plt.figure()
plt.plot(t, epsilons)
plt.show()

#%% Plot alphas
plt.figure()
plt.plot(t, alphas)
plt.show()
#%%