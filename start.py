"""
Kernel Methods Data Challenge - Full Pipeline

Best config: CKN (patch_size=3, n_filters=256, subsampling=2) + KRR (poly, reg=0.8, degree=4, coef0=6)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from CKN.utils import extract_patches, normalize_patches, spherical_kmeans, ckn_activation, gaussian_pooling, optimize_W_and_eta
from CKN.layer import CKNLayer
from CKN.network import CKNNetwork
from classifier.KRR import KRRClassifier
from classifier.SVM import ClassicSVM, MultiClassClassicSVM
from kernels import polynomial_kernel, linear_kernel, rbf_kernel
from data import load_data, split_data_train_val
from augment import augment_flip, augment_shift
from utils import get_accuracy, generate_submission


FILE_PATH = './data'

X_train_raw, X_test_raw, y_train = load_data(FILE_PATH)

print(f"X_train shape : {X_train_raw.shape}")
print(f"X_test  shape : {X_test_raw.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"Classes : {np.unique(y_train)}")


X_tr, X_val, y_tr, y_val = split_data_train_val(X_train_raw, y_train, val_size=0.1, seed=42)

print(f"\nTrain : {X_tr.shape[0]} images")
print(f"Validation : {X_val.shape[0]} images")

def prepare_data(X_raw, y=None, augment=False):
    N = X_raw.shape[0]
    X = X_raw.reshape(N, 3, 32, 32)

    if augment and y is not None:
        X, y = augment_flip(X, y)
        X, y = augment_shift(X, y, max_shift=2)

    X_scaled = X / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std  = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
    X_norm = (X_scaled - mean) / std

    if y is not None:
        return X_norm, y
    return X_norm


X_train_imgs, y_train_ready = prepare_data(X_tr, y_tr, augment=True)
X_val_imgs, y_val_ready = prepare_data(X_val, y_val,augment=False)
X_test_imgs = prepare_data(X_test_raw,augment=False)

print(f"\nTrain final : {X_train_imgs.shape}")
print(f"Val final : {X_val_imgs.shape}")
print(f"Test final : {X_test_imgs.shape}")

X_train_imgs = X_train_imgs.reshape(X_train_imgs.shape[0], 3, 32, 32)
X_val_imgs = X_val_imgs.reshape(X_val_imgs.shape[0], 3, 32, 32)
X_test_imgs = X_test_imgs.reshape(X_test_imgs.shape[0], 3, 32, 32)

layers_config = [
    {'patch_size': 3, 'n_filters': 256, 'subsampling': 2, 'sigma': None},
]

ckn = CKNNetwork(layers_config)
ckn.unsup_train_all(list(X_train_imgs), max_patches=100_000, n_pairs=2_000)

print("\nExtracting features train...")
F_train = ckn.extract_features(list(X_train_imgs))

print("Extracting features validation...")
F_val = ckn.extract_features(list(X_val_imgs))

print("Extracting features test...")
F_test = ckn.extract_features(list(X_test_imgs))


F_train = F_train / (np.linalg.norm(F_train, axis=1, keepdims=True) + 1e-8)
F_val   = F_val   / (np.linalg.norm(F_val,   axis=1, keepdims=True) + 1e-8)
F_test  = F_test  / (np.linalg.norm(F_test,  axis=1, keepdims=True) + 1e-8)

print(f"\nFeature dimension : {F_train.shape[1]}")

krr = KRRClassifier(polynomial_kernel, reg=0.8, num_classes=10, degree=4, coef0=6)
krr.fit(F_train, y_train_ready)

predictions = krr.predict(F_val)
acc = get_accuracy(predictions, y_val)
print(f"\nAccuracy on validation set: {acc:.2f}%")

pred_test = krr.predict(F_test)
generate_submission(pred_test, model_name="krr_ckn_poly")
print("Submission file saved in ./submissions/")