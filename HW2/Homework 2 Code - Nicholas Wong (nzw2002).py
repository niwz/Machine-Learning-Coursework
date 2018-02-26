#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from scipy.special import expit

#%% Loading the Data

X_train = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW2/hw2-data/X_train.csv', header = None)
y_train = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW2/hw2-data/y_train.csv', header = None)
X_test = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW2/hw2-data/X_test.csv', header = None)
y_test = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW2/hw2-data/y_test.csv', header = None)

#%% Convert to numpy arrays

X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

#%%

class NaiveBayes(object):
    
    def parameters(self, X, y):
        
        # Calculate Pi
        self.pi = np.mean(y)
                
        # Create boolean vectors for subsetting the data
        ham = np.array([y == 0 for y in y.reshape(1, -1).tolist()[0]])
        spam = np.invert(ham)
        
        # Split the training data
        X_bernoulli = X[:, :-3]
        X_pareto = X[:, -3:]
        
        # Create dictionary for theta parameters        
        self.theta1 = {
                'ham': np.apply_along_axis(np.mean, 0, X_bernoulli[ham, ]),
                'spam': np.apply_along_axis(np.mean, 0, X_bernoulli[spam, ])
                }
        
        self.theta2 = {
                'ham': np.array([len(X_pareto[ham, i]) / sum(np.log(X_pareto[ham, i])) for i in range(3)]),
                'spam': np.array([len(X_pareto[spam, i]) / sum(np.log(X_pareto[spam, i])) for i in range(3)])
                }
    
    def predict(self, X):
        
        # Create matrix of predictions
        self.predictions = np.zeros(shape = (X.shape[0], 1))
        
        # Calculate posterior for ham
        ll_ham_bernoulli = np.zeros(shape = (X.shape[0], 54))
        ll_ham_pareto = np.zeros(shape = (X.shape[0], 3))
        for i in range(X.shape[0]):
            for j in range(54):
                ll_ham_bernoulli[i, j] = X[i, j] * np.log(self.theta1['ham'][j]) + (1 - X[i, j]) * np.log(1 - self.theta1['ham'][j])
            for j in range(3):
                ll_ham_pareto[i, j] = np.log(self.theta2['ham'][j]) - (self.theta2['ham'][j] + 1) * np.log(X[i, j+54])
        prob_ham = ll_ham_bernoulli.sum(axis = 1) + ll_ham_pareto.sum(axis = 1) + np.log(self.pi)
        
        # Calculate posterior for spam
        ll_spam_bernoulli = np.zeros(shape = (X.shape[0], 54))
        ll_spam_pareto = np.zeros(shape = (X.shape[0], 3))
        for i in range(X.shape[0]):
            for j in range(54):
                ll_spam_bernoulli[i, j] = X[i, j] * np.log(self.theta1['spam'][j]) + (1 - X[i, j]) * np.log(1 - self.theta1['spam'][j])
            for j in range(3):
                ll_spam_pareto[i, j] = np.log(self.theta2['spam'][j]) - (self.theta2['spam'][j] + 1) * np.log(X[i, j+54])
        prob_spam = ll_spam_bernoulli.sum(axis = 1) + ll_spam_pareto.sum(axis = 1) + np.log(1 - self.pi)
        
        # Calculate predictions
        for i in range(X.shape[0]):
            if prob_spam[i] > prob_ham[i]:
                self.predictions[i] = 1
        
    def xtab(self, y):
        self.confusion_matrix = np.zeros(shape = (2, 2))
        for i in range(2):
            for j in range(2):
                self.confusion_matrix[i, j] = sum([(self.predictions[k] == i) & (y[k] == j) for k in range(93)])
        print(self.confusion_matrix)
        
    def accuracy(self):
        print((self.confusion_matrix[0, 0] + self.confusion_matrix[1, 1]) / 93)
        
    def plot(self):
        fig, ax = plt.subplots(2,1)
        ax1, ax2 = ax.ravel()
        markerline, stemlines, baseline = ax1.stem(range(1,55), model.theta1['ham'], '-')
        plt.setp(markerline, 'markerfacecolor', 'g')
        plt.setp(baseline, visible=False)
        ax1.set_title('Parameters for y = 0')
        
        markerline, stemlines, baseline = ax2.stem(range(1,55), model.theta1['spam'], '-')
        plt.setp(markerline, 'markerfacecolor', 'r')
        plt.setp(baseline, visible=False)
        ax2.set_title('Parameters for y = 1')
        
        plt.tight_layout()
        plt.show()
        
#%% Output for Naive Bayes classifier:
                
model = NaiveBayes()
model.parameters(X_train, y_train)
model.predict(X_test)
model.xtab(y_test)
model.accuracy()
model.plot()

#%% k-nearest neighbors

class kNN(object):
        
    def fit(self, X_train, X_test, y_train):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
#        for i in range(54, 57):
#            mu = np.mean(self.X_train[:,i])
#            sigma = np.std(self.X_train[:,i])
#            self.X_train[:, i] = (self.X_train[:, i] - mu) / sigma
#            self.X_test[:, i] = (self.X_test[:, i] - mu) / sigma
        self.labels = y_train.copy()[:, 0]
    
    def predict(self, k):
        
        predictions = np.zeros((self.X_test.shape[0], 1))
        ''' Iterate over the number of rows in the testing data so that 
        by the end of this chunk of code each row of the testing data will have
        its k nearest neighbors defined.'''
        
        for i in range(self.X_test.shape[0]): 
            
            # Create a sorted list to store tuples of (distance, label) for knn
            dist = SortedList() 
            
            # Iterate over the rows of the training data
            for j in range(self.X_train.shape[0]): 
                
                # Find the distance between row i of training and row j of testing
                d = np.sum(abs(self.X_train[j, :] - X_test[i, :]))
                
                # Check if the list of nearest neighbors is less than k
                if len(dist) < k:
                    dist.add((d, self.labels[j]))
                
                # Check if the current value of d is smaller than that of any existing nn
                else:
                    if d < dist[-1][0]:
                        del dist[-1]
                        dist.add((d, self.labels[j]))
            
            # Find out which label is the majority among nearest neighbors
            nearest = {}
            for _, n in dist:
                nearest[n] = nearest.get(n, 0) + 1
                
            max_votes = 0
            max_votes_class = -1
            for c, votes in dict.items(nearest):
                if votes > max_votes:
                    max_votes = votes
                    max_votes_class = c
            predictions[i] = max_votes_class
            
        return predictions
     
    def accuracy(self, actual, k_lim):
        P = [self.predict(k) for k in range(1, k_lim + 1)]
        self.scores = [np.mean(P[i] == actual) for i in range(k_lim)]
        
    def plot(self, k_lim):
        plt.plot(range(1, k_lim + 1), self.scores)
        plt.show()

        
    
#%% Output for knn

model2 = kNN()
model2.fit(X_train, X_test, y_train)
model2.accuracy(y_test, 20)
model2.plot(20)

#%% Logit class

class logit(object):
  
    
    def __init__(self, X_train, y_train):
        self.X_train = np.column_stack((X_train, np.ones(X_train.shape[0])))
        self.y_train = y_train.copy()
        self.y_train[self.y_train == 0] = -1
        
           
    def ascent(self, iterations = 10000, smoothing = 1e-10):
        
        # Initialize loss function and weights as empty vectors
        L = []
        weights = np.zeros(self.X_train.shape[1]).reshape(-1,1)
        
        for t in range(1, iterations + 1):
            
            # Calculate and store the loss function output
            eta = 1 / (1e5 * np.sqrt(t + 1))
            activation = expit(np.multiply(self.y_train, self.X_train.dot(weights)))
            L.append(np.sum(np.log(activation + smoothing)))
            
            # Update the weights
            deriv = np.dot(self.X_train.T, np.multiply(self.y_train, 1 - activation))
            weights = weights + eta * deriv
            
        return L, weights
            
    
    def newton(self, iterations = 100, smoothing = 1e-10):
        
        # Initialize loss function and weights as empty vectors
        L = []
        weights = np.zeros(self.X_train.shape[1]).reshape(-1,1)
        
        for t in range(1, iterations + 1):
            
            # Calculate and store the loss function output
            eta = 1 / (np.sqrt(t + 1))
            activation = expit(np.multiply(self.y_train, self.X_train.dot(weights)))
            L.append(np.sum(np.log(activation + smoothing)))
            
            # Update the weights
            deriv = np.dot(self.X_train.T, np.multiply(self.y_train, 1 - activation))
            deriv2 = - np.multiply(np.multiply(activation ,1 - activation), self.X_train).T.dot(self.X_train)
            weights = weights - eta * np.dot(np.linalg.inv(deriv2), deriv)
            
        return L, weights
    
    def predict(self, X, ascent = False, newton = False):
         
        X = np.column_stack((X, np.ones(X.shape[0])))
        
        if ascent:
            _, weights = self.ascent()
            predictions = np.sign(np.dot(X, weights))
            return predictions
        
        if newton:
            _, weights = self.newton()
            predictions = np.sign(np.dot(X, weights))
            return predictions
        
    def accuracy(self, y, X, ascent = False, newton = False):
        y = y.copy()
        y[y == 0] = -1
        if ascent:
            predictions = self.predict(X, ascent = True)
        if newton:
            predictions = self.predict(X, newton = True)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def plot(self, ascent = False, newton = False):
        
        if ascent:
            L, _ = self.ascent()
            plt.plot(L)
            plt.show()
            
        if newton:
            L, _ = self.newton()
            plt.plot(L)
            plt.show()
        
#%%

model3 = logit(X_train, y_train)
model3.plot(ascent = True)
model3.plot(newton = True)
model3.accuracy(y_test, X_test, newton = True)
#%%


