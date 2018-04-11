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
