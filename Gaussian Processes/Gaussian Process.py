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