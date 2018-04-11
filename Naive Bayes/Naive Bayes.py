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