import numpy as np
import matplotlib.pyplot as plt

#%% Generate the data

mu1, mu2, mu3 = [0, 0], [3, 0], [0, 3]
cov = np.eye(2)

x1 = np.random.multivariate_normal(mu1, cov, (100))
x2 = np.random.multivariate_normal(mu2, cov, (250))
x3 = np.random.multivariate_normal(mu3, cov, (150))
x = np.vstack((x1, x2, x3))
#%%
class KMeans(object):
    
    def __init__(self, k, x):
        self.k = k
        self.x = x.copy()
        np.random.shuffle(self.x)
        self.centroids = self.x[:self.k]
        self.objective = []
        
    def get_cluster(self, obs):
        dist = np.sum((self.centroids - obs)**2, axis = 1)
        label = np.argmin(dist)
        error = dist[label]
        return (label, error)
        
    
    def fit(self, iterations = 20):
        
        for i in range(iterations):    
            
            self.cluster  = np.apply_along_axis(self.get_cluster, 1, self.x)
            self.objective.append(np.sum(self.cluster[:,1]))
            for k in range(self.k):
                self.centroids[k,:] = np.mean(self.x[self.cluster[:,0] == k], axis = 0)
                
    def plot(self):
        
        plt.scatter(self.x[:,0], self.x[:,1], c = self.cluster[:,0] , cmap = 'Paired')
        plt.scatter(self.centroids[:,0], self.centroids[:,1], c = 'r', s = 100)
        plt.title('Clusters for K = {}'.format(self.k))
        plt.show()
        
#%% Plot clusters
K_values = [2, 3, 4, 5]
objective = []
for k in K_values:
    km = KMeans(k, x)
    km.fit(20)
    objective.append(km.objective)
    if k in [3, 5]:
        km.plot()
        
#%% Plot value of objective function
plt.figure()
plt.plot(np.arange(1, 21), objective[0], label = 'k = 2')
plt.plot(np.arange(1, 21), objective[1], label = 'k = 3')
plt.plot(np.arange(1, 21), objective[2], label = 'k = 4')
plt.plot(np.arange(1, 21), objective[3], label = 'k = 5')
plt.xticks(np.arange(1, 21, 1.0))
plt.legend(loc = 'upper right')
plt.xlabel('Number of Iterations')
plt.ylabel('Objective Function Value')
plt.title('Objective vs Iteration for K = [2,3,4,5]')
plt.show()