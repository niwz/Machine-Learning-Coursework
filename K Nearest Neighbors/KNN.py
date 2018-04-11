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
