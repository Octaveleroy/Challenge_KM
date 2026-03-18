"""
preprocessing.py — Incremental preprocessing pipelines for CKN on CIFAR-10
============================================================================
V1: baseline (reshape + /255 + channel z-score)
V2: + ZCA whitening (replaces channel z-score)
V3: + local contrast normalization (per-patch, inside CKN forward pass)
V4: + augmentation toggle (for ablation)

Each version is a standalone class with fit/transform interface.
"""

import numpy as np


# ── Utility: Local Contrast Normalization ──────────────────────────────

def local_contrast_normalize(patches, epsilon=1e-6):
    """
    Per-patch local contrast normalization.

    Removes the DC component (patch mean) so the unit-sphere projection
    captures edge/texture direction rather than local brightness.
    Preserves contrast magnitude as a separate scalar weight.

    For each patch x:  x̃ = (x - x̄) / ‖x - x̄‖₂,  c = ‖x - x̄‖₂

    :param patches: (N, patch_dim)
    :param epsilon: floor for zero-contrast patches
    :return: (normalized_patches, contrast_norms)
    """
    patch_means = patches.mean(axis=1, keepdims=True)
    centered = patches - patch_means
    contrast_norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(contrast_norms, epsilon)
    return normalized, contrast_norms


# ── Utility: Data Augmentation ─────────────────────────────────────────

def augment_images(images, max_shift=2):
    """
    Stochastic augmentation: random horizontal flip + random translation.

    :param images: (N, C, H, W) array
    :param max_shift: maximum pixel shift in each direction
    :return: (N, C, H, W) augmented array
    """
    N, C, H, W = images.shape
    aug = np.empty_like(images)

    for i in range(N):
        img = images[i]
        if np.random.random() < 0.5:
            img = img[:, :, ::-1].copy()

        dy = np.random.randint(-max_shift, max_shift + 1)
        dx = np.random.randint(-max_shift, max_shift + 1)
        padded = np.pad(img,
                        ((0, 0), (max_shift, max_shift), (max_shift, max_shift)),
                        mode='constant')
        y0, x0 = max_shift + dy, max_shift + dx
        aug[i] = padded[:, y0:y0 + H, x0:x0 + W]

    return aug


# ── V1: Baseline ──────────────────────────────────────────────────────

class PreprocessV1:
    """
    Baseline: reshape → [0,1] → channel-wise z-score.

    Channel z-score removes global brightness/color bias but does NOT
    decorrelate spatial correlations within or across channels.
    """

    def __init__(self):
        self.channel_means = None
        self.channel_stds = None

    def fit(self, X_flat):
        images = X_flat.reshape(-1, 3, 32, 32) / 255.0
        self.channel_means = images.mean(axis=(0, 2, 3))
        self.channel_stds = images.std(axis=(0, 2, 3))
        self.channel_stds[self.channel_stds < 1e-8] = 1.0
        return self

    def transform(self, X_flat):
        images = X_flat.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        for c in range(3):
            images[:, c] = (images[:, c] - self.channel_means[c]) / self.channel_stds[c]
        return [images[i] for i in range(len(images))]

    @property
    def use_lcn(self):
        return False

    def __repr__(self):
        return "PreprocessV1(channel_zscore)"


# ── V2: + ZCA Whitening ───────────────────────────────────────────────

class PreprocessV2:
    """
    ZCA whitening: X_zca = U (Λ + εI)^{-1/2} U^T (X - μ).

    Decorrelates all pixel pairs (spatial + cross-channel) while preserving
    spatial arrangement. ε regularizes low-variance directions.
    """

    def __init__(self, eps=0.1):
        self.eps = eps
        self.mean = None
        self.W_zca = None

    def fit(self, X_flat):
        X = X_flat.astype(np.float64) / 255.0
        self.mean = X.mean(axis=0).astype(np.float32)

        X_centered = X - self.mean
        cov = (X_centered.T @ X_centered) / X_centered.shape[0]
        lam, U = np.linalg.eigh(cov)
        lam = np.maximum(lam, 1e-12)

        self.W_zca = (U * (1.0 / np.sqrt(lam + self.eps))[np.newaxis, :]) @ U.T
        self.W_zca = self.W_zca.astype(np.float32)

        print(f"  ZCA: top eigenvalue={lam.max():.6f}, "
              f"bottom={lam.min():.6f}, eps={self.eps}")
        return self

    def transform(self, X_flat):
        X = X_flat.astype(np.float32) / 255.0
        X_white = (X - self.mean) @ self.W_zca
        images = X_white.reshape(-1, 3, 32, 32).astype(np.float32)
        return [images[i] for i in range(len(images))]

    @property
    def use_lcn(self):
        return False

    def __repr__(self):
        return f"PreprocessV2(ZCA, eps={self.eps})"


# ── V3: + Local Contrast Normalization ─────────────────────────────────

class PreprocessV3:
    """
    ZCA (from V2) + local contrast normalization on patches.

    Image-level preprocessing is identical to V2. use_lcn=True signals
    the CKN layer to apply local_contrast_normalize before kernel computation.
    This removes per-patch DC so the kernel captures pure edge/texture.
    """

    def __init__(self, eps=0.1):
        self._v2 = PreprocessV2(eps=eps)

    def fit(self, X_flat):
        self._v2.fit(X_flat)
        return self

    def transform(self, X_flat):
        return self._v2.transform(X_flat)

    @property
    def use_lcn(self):
        return True

    def __repr__(self):
        return f"PreprocessV3(ZCA+LCN, eps={self._v2.eps})"


# ── V4: + Augmentation flag ───────────────────────────────────────────

class PreprocessV4:
    """
    ZCA + LCN (from V3) + explicit augmentation toggle for ablation.
    """

    def __init__(self, eps=0.1, augment=False, max_shift=2):
        self._v3 = PreprocessV3(eps=eps)
        self.augment = augment
        self.max_shift = max_shift

    def fit(self, X_flat):
        self._v3.fit(X_flat)
        return self

    def transform(self, X_flat):
        return self._v3.transform(X_flat)

    def augment_batch(self, images_array):
        """Apply stochastic augmentation to (N, C, H, W) if enabled."""
        if self.augment:
            return augment_images(images_array, max_shift=self.max_shift)
        return images_array

    @property
    def use_lcn(self):
        return True

    def __repr__(self):
        return (f"PreprocessV4(ZCA+LCN, eps={self._v3._v2.eps}, "
                f"augment={self.augment})")


# ── Comparison function ───────────────────────────────────────────────

def compare_versions(X_train_flat, y_train, X_val_flat, y_val,
                     ckn_config=None, krr_config=None, seed=42):
    """
    Run all four preprocessing pipelines through identical CKN + KRR.

    :param ckn_config: list of dicts for CKNLayer (patch_size, n_filters, subsampling, sigma)
    :param krr_config: dict with 'kernel_func', 'reg', and any extra kernel_params
    :return: dict {version_name: {'train_acc': float, 'val_acc': float}}
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from CKN.network import CKNNetwork
    from classifier.KRR import KRRClassifier
    from kernels import polynomial_kernel

    if ckn_config is None:
        ckn_config = [
            {'patch_size': 3, 'n_filters': 256, 'subsampling': 2, 'sigma': None},
            {'patch_size': 3, 'n_filters': 256, 'subsampling': 1, 'sigma': None},
        ]
    if krr_config is None:
        krr_config = {
            'kernel_func': polynomial_kernel,
            'reg': 100.0,
            'degree': 3,
            'coef0': 1.0,
        }

    versions = {
        'V1_baseline': PreprocessV1(),
        'V2_zca': PreprocessV2(eps=0.1),
        'V3_zca_lcn': PreprocessV3(eps=0.1),
        'V4_zca_lcn_noaug': PreprocessV4(eps=0.1, augment=False),
    }

    results = {}

    for name, preprocessor in versions.items():
        print(f"\n{'='*60}")
        print(f"  {name}: {preprocessor}")
        print(f"{'='*60}")

        np.random.seed(seed)
        preprocessor.fit(X_train_flat)
        train_images = preprocessor.transform(X_train_flat)
        val_images = preprocessor.transform(X_val_flat)

        network = CKNNetwork(ckn_config)
        network.unsup_train_all(train_images, max_patches=100000, n_pairs=5000)

        X_tr_feat = network.extract_features(train_images)
        X_val_feat = network.extract_features(val_images)

        # L2-normalize each sample so x^T x = 1 → polynomial kernel stays bounded
        mu = X_tr_feat.mean(0)
        X_tr_feat = X_tr_feat - mu
        X_val_feat = X_val_feat - mu
        tr_norms = np.linalg.norm(X_tr_feat, axis=1, keepdims=True)
        tr_norms[tr_norms < 1e-8] = 1.0
        X_tr_feat = X_tr_feat / tr_norms
        val_norms = np.linalg.norm(X_val_feat, axis=1, keepdims=True)
        val_norms[val_norms < 1e-8] = 1.0
        X_val_feat = X_val_feat / val_norms

        kernel_func = krr_config['kernel_func']
        kernel_params = {k: v for k, v in krr_config.items()
                         if k not in ('kernel_func', 'reg')}
        clf = KRRClassifier(
            kernel_func=kernel_func, reg=krr_config['reg'], **kernel_params,
        )
        clf.fit(X_tr_feat, y_train.astype(int))

        train_preds = clf.predict(X_tr_feat)
        val_preds = clf.predict(X_val_feat)
        train_acc = np.mean(train_preds == y_train.astype(int))
        val_acc = np.mean(val_preds == y_val.astype(int))
        results[name] = {'train_acc': train_acc, 'val_acc': val_acc}
        print(f"  → train={train_acc:.4f}, val={val_acc:.4f}")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Version':<25} {'Train':>8} {'Val':>8}")
    print(f"  {'-'*41}")
    for name, acc in results.items():
        print(f"  {name:<25} {acc['train_acc']:>8.4f} {acc['val_acc']:>8.4f}")

    return results


# ── Main: quick comparison with small CKN ────────────────────────────

if __name__ == '__main__':
    import sys
    import os
    import time

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data import load_data, split_data_train_val
    from kernels import polynomial_kernel

    # ── Load & split ──────────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    X_train_full, _, y_train_full = load_data(data_path)
    X_train, X_val, y_train, y_val = split_data_train_val(
        X_train_full, y_train_full, val_size=0.2, seed=42,
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # ── Small CKN config (fast iteration) ─────────────────────────────
    small_ckn_config = [
        {'patch_size': 3, 'n_filters': 32, 'subsampling': 2, 'sigma': None},
        {'patch_size': 3, 'n_filters': 32, 'subsampling': 2, 'sigma': None},
    ]
    krr_config = {
        'kernel_func': polynomial_kernel,
        'reg': 10.0,
        'degree': 3,
        'coef0': 1.0,
    }

    print("\n" + "=" * 60)
    print("  SMALL CKN PREPROCESSING COMPARISON  (n_filters=32)")
    print("=" * 60)

    t0 = time.time()
    results = compare_versions(
        X_train, y_train, X_val, y_val,
        ckn_config=small_ckn_config, krr_config=krr_config, seed=42,
    )
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
