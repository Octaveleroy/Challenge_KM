import numpy as np

class KRRClassifier:
    """ Kerner Ridge Regression""" 
    def __init__(self, kernel_func ,reg= 0.1,num_classes = 10, **kernel_params):
        self.kernel_func = kernel_func
        self.reg = reg
        self.kernel_params = kernel_params
        self.alpha = None
        self.X_train = None
        self.num_classes = num_classes

    def fit(self,X,y) : 
        self.X_train = X
        n = X.shape[0]
        K = self.kernel_func(X,X,**self.kernel_params)
        
        # One hot encoding matrix for classes
        Y_one_hot = np.zeros((n, self.num_classes))
        Y_one_hot[np.arange(n), y.astype(int)] = 1

        self.alpha = np.linalg.solve(K + self.reg * np.eye(n), Y_one_hot)
        
    def predict(self,X_test) : 
        K_test = self.kernel_func(X_test,self.X_train,**self.kernel_params)
        scores = np.dot(K_test, self.alpha)
        return np.argmax(scores, axis=1)
