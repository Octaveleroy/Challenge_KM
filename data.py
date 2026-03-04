import pandas as pd
import numpy as np


def load_data(file_path : str) -> tuple:
    """
    Load data of the file path

    :param file_path: Path to the data directory
    :type file_path: str
    """
    print("Load data")
    X_train = pd.read_csv(file_path + '/Xtr.csv', header=None, usecols=range(3072)).values.astype('float32')
    X_test = pd.read_csv(file_path + '/Xte.csv', header=None, usecols=range(3072)).values.astype('float32')
    y_train = pd.read_csv(file_path + '/Ytr.csv', usecols=[1]).values.ravel()
    
    return X_train, X_test, y_train


def split_data_train_val(X,y,val_size = 0.2,seed=42):
    """
    Split the data in a training and validation sets

    :param X: Data to split 
    :param y: Data's label to split
    :param val_size: Ratio of data to put in validation
    :param seed: seed of the split, to have reproductible experiments 
    """
    np.random.seed(seed)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_idx = int(X.shape[0]*(1 -val_size))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
