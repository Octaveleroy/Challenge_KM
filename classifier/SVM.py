
import numpy as np
import cvxopt

cvxopt.solvers.options['show_progress'] = False

class ClassicSVM:
    """
    SVM for classification
    """
    def __init__(self, C=1.0):
        self.C = C
        self.alphas = None
        self.b = 0.0
        self.sv_indices = None

    def fit(self, K, Y):
        n = len(Y)

        y = (2 * Y - 1).astype(float) if set(Y) == {0, 1} else Y.astype(float)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n))

        A = cvxopt.matrix(y, (1, n), 'd')
        b_eq = cvxopt.matrix(0.0)

        G = cvxopt.matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        solver = cvxopt.solvers.qp(P, q, G, h, A, b_eq)
        alphas_full = np.ravel(solver['x'])

        eps = 1e-5
        self.sv_indices = alphas_full > eps

        self.alpha = (alphas_full * y)
        self.alpha[~self.sv_indices] = 0.0

        margin_indices = (alphas_full > eps) & (alphas_full < self.C - eps)

        if np.any(margin_indices):
            b_list = []
            for i in np.where(margin_indices)[0]:
                b_i = y[i] - np.sum(self.alpha * K[:, i])
                b_list.append(b_i)
            self.b = np.mean(b_list)
        else:
            self.b = 0.0

    def decision_function(self, K_test):
        return np.dot(K_test, self.alpha) + self.b

    def predict(self, K_test):
        return np.sign(self.decision_function(K_test))

class MultiClassClassicSVM:
    """
    Mutli Class One-vs-Rest type for SVM. To do multiclass predictions
    """
    def __init__(self, num_classes=10, C=1.0, kernel='rbf', gamma=0.1):
        self.num_classes = num_classes
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.X_train = None 
        self.classifiers = [ClassicSVM(C=self.C) for _ in range(num_classes)]

    def _compute_kernel(self, X1, X2):
        if self.kernel_type == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel_type == 'rbf':
            sq_dists = (np.sum(X1**2, axis=1).reshape(-1, 1) +
                        np.sum(X2**2, axis=1) -
                        2 * np.dot(X1, X2.T))
            return np.exp(-self.gamma * sq_dists)

    def fit(self, X, y):
        self.X_train = X
        K_train = self._compute_kernel(X, X)

        for c in range(self.num_classes):
            y_binary = np.where(y == c, 1, -1)
            self.classifiers[c].fit(K_train, y_binary)

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        scores = np.zeros((n_samples, self.num_classes))

        K_test = self._compute_kernel(X_test, self.X_train)
        
        for c in range(self.num_classes):
            scores[:, c] = self.classifiers[c].decision_function(K_test)

        return np.argmax(scores, axis=1)