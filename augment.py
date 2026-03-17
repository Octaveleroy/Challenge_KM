import numpy as np


def augment_flip(X, y):
    """
    Flip the images to augment it 
    """
    X_flipped = X[:, :, :, ::-1] 
    X_augmented = np.concatenate((X, X_flipped), axis=0)
    y_augmented = np.concatenate((y, y), axis=0)
    return X_augmented, y_augmented


def augment_shift(X, y, max_shift=2):
    """
    Add little shift on the images to augment
    """
    N, C, H, W = X.shape
    X_shifted = np.copy(X)
    for i in range(N):
        shift_h = np.random.randint(-max_shift, max_shift + 1)
        shift_w = np.random.randint(-max_shift, max_shift + 1)
        X_shifted[i] = np.roll(X_shifted[i], (shift_h, shift_w), axis=(1, 2))
    
    X_augmented = np.concatenate((X, X_shifted), axis=0)
    y_augmented = np.concatenate((y, y), axis=0)
    return X_augmented, y_augmented