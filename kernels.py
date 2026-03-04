import numpy as np

def rbf_kernel(X1, X2, gamma=0.01):
    """
    Compute a RBF kernel

    :param X1: first points to compute the kernel
    :param X2: second points to compute the kernel
    :return: array of the kernels computed
    """
    sq_norm_X1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    sq_norm_X2 = np.sum(X2**2, axis=1)
    distances = sq_norm_X1 + sq_norm_X2 - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * distances)

def linear_kernel(X1, X2):
    """
    Compute Linear Kernel

    :param X1: first points to compute the kernel
    :param X2: second points to compute the kernel
    """
    return np.dot(X1, X2.T)

def polynomial_kernel(X1, X2, degree=2, coef0=0.05):
    """
    Compute the polynomial kernel

    :param X1: first points to compute the kernel
    :param X2: second points to compute the kernel
    :degree: Degree of the polynome to calculate
    :coef0: coef of the bias
    """
    return (np.dot(X1, X2.T) + coef0) ** degree
