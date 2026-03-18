"""
ensemble.py — Ensembling strategies for CKN + KRR on CIFAR-10
==============================================================
Strategies:
  1. Majority voting       — hard labels, equal weight
  2. Soft voting            — average continuous scores, then argmax
  3. Weighted soft voting   — val-accuracy-weighted score averaging
  4. Stacking (KRR meta)   — train a second-level KRR on base model scores
  5. TTA ensemble           — test-time augmentation + soft voting

All strategies expect base models that expose:
  - predict(X)        → (N,) int labels
  - predict_scores(X) → (N, C) float scores
"""

import numpy as np
from classifier.KRR import KRRClassifier
from kernels import polynomial_kernel, rbf_kernel, linear_kernel


# ── Strategy 1: Majority Voting ──────────────────────────────────────

def majority_vote(predictions_list, num_classes=10):
    """
    Hard majority voting over a list of prediction arrays.

    :param predictions_list: list of (N,) int arrays
    :return: (N,) int — most frequent label per sample
    """
    N = predictions_list[0].shape[0]
    votes = np.zeros((N, num_classes), dtype=int)
    for preds in predictions_list:
        for c in range(num_classes):
            votes[:, c] += (preds == c)
    return np.argmax(votes, axis=1)


# ── Strategy 2: Soft Voting ──────────────────────────────────────────

def soft_vote(scores_list):
    """
    Average continuous scores across models, then argmax.

    :param scores_list: list of (N, C) float arrays
    :return: (N,) int predictions
    """
    avg_scores = np.mean(scores_list, axis=0)
    return np.argmax(avg_scores, axis=1)


# ── Strategy 3: Weighted Soft Voting ─────────────────────────────────

def weighted_soft_vote(scores_list, weights):
    """
    Weighted average of continuous scores.

    :param scores_list: list of (N, C) float arrays
    :param weights: list of floats (e.g. validation accuracies)
    :return: (N,) int predictions
    """
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    weighted_scores = sum(w * s for w, s in zip(weights, scores_list))
    return np.argmax(weighted_scores, axis=1)


# ── Strategy 4: Stacking ─────────────────────────────────────────────

def stacking(scores_train_list, y_train, scores_test_list,
             reg=1.0, kernel_func=None, **kernel_params):
    """
    Level-2 KRR trained on concatenated base-model scores.

    :param scores_train_list: list of (N_train, C) score arrays
    :param y_train: (N_train,) true labels
    :param scores_test_list: list of (N_test, C) score arrays
    :return: (N_test,) int predictions
    """
    if kernel_func is None:
        kernel_func = linear_kernel
        kernel_params = {}

    meta_train = np.hstack(scores_train_list)  # (N_train, C * n_models)
    meta_test = np.hstack(scores_test_list)

    meta_clf = KRRClassifier(
        kernel_func=kernel_func, reg=reg, **kernel_params,
    )
    meta_clf.fit(meta_train, y_train.astype(int))
    return meta_clf.predict(meta_test)


# ── Strategy 5: TTA Ensemble ─────────────────────────────────────────

def tta_augment(images, max_shift=2):
    """
    Generate TTA views: original + hflip + 4 directional shifts.

    :param images: (N, C, H, W) array
    :return: list of (N, C, H, W) arrays (6 views)
    """
    _, _, H, W = images.shape
    views = [images]

    # horizontal flip
    views.append(images[:, :, :, ::-1].copy())

    # 4 cardinal shifts
    for dy, dx in [(-max_shift, 0), (max_shift, 0),
                   (0, -max_shift), (0, max_shift)]:
        shifted = np.roll(images, (dy, dx), axis=(2, 3))
        views.append(shifted)

    return views


# ── Full comparison harness ──────────────────────────────────────────

def build_diverse_models(kernel_configs=None):
    """
    Return a list of (name, KRRClassifier) with diverse kernel settings.

    Default: 5 models spanning polynomial degrees and an RBF.
    """
    if kernel_configs is None:
        kernel_configs = [
            ('poly_d3_c1',  dict(kernel_func=polynomial_kernel, reg=10.0,
                                 degree=3, coef0=1.0)),
            ('poly_d4_c10', dict(kernel_func=polynomial_kernel, reg=10.0,
                                 degree=4, coef0=10.0)),
            ('poly_d3_c10', dict(kernel_func=polynomial_kernel, reg=10.0,
                                 degree=3, coef0=10.0)),
            ('poly_d5_c10', dict(kernel_func=polynomial_kernel, reg=10.0,
                                 degree=5, coef0=10.0)),
            ('rbf_g001',    dict(kernel_func=rbf_kernel, reg=10.0,
                                 gamma=0.01)),
        ]

    models = []
    for name, cfg in kernel_configs:
        kf = cfg.pop('kernel_func')
        reg = cfg.pop('reg')
        models.append((name, KRRClassifier(kernel_func=kf, reg=reg, **cfg)))
    return models


def run_ensemble_comparison(X_train, y_train, X_val, y_val,
                            models=None, seed=42):
    """
    Train diverse KRR models on shared features, then compare all
    ensembling strategies against individual baselines.

    :param X_train: (N_train, D) feature matrix (already extracted by CKN)
    :param y_train: (N_train,) int labels
    :param X_val: (N_val, D) feature matrix
    :param y_val: (N_val,) int labels
    :return: dict {strategy_name: val_accuracy}
    """
    np.random.seed(seed)

    if models is None:
        models = build_diverse_models()

    # ── Train all base models ─────────────────────────────────────
    print(f"Training {len(models)} base models...")
    val_accs = []
    preds_list = []
    scores_val_list = []
    scores_train_list = []

    for name, clf in models:
        clf.fit(X_train, y_train.astype(int))

        scores_tr = clf.predict_scores(X_train)
        scores_val = clf.predict_scores(X_val)
        preds = np.argmax(scores_val, axis=1)

        acc = np.mean(preds == y_val.astype(int))
        val_accs.append(acc)
        preds_list.append(preds)
        scores_val_list.append(scores_val)
        scores_train_list.append(scores_tr)

        print(f"  {name:<20s}  val_acc={acc:.4f}")

    # ── Ensemble strategies ───────────────────────────────────────
    results = {}

    # Individual baselines
    for i, (name, _) in enumerate(models):
        results[f'base_{name}'] = val_accs[i]

    # 1. Majority vote
    mv_preds = majority_vote(preds_list)
    results['majority_vote'] = np.mean(mv_preds == y_val.astype(int))

    # 2. Soft vote
    sv_preds = soft_vote(scores_val_list)
    results['soft_vote'] = np.mean(sv_preds == y_val.astype(int))

    # 3. Weighted soft vote (weights = val accuracies)
    wsv_preds = weighted_soft_vote(scores_val_list, val_accs)
    results['weighted_soft_vote'] = np.mean(wsv_preds == y_val.astype(int))

    # 4. Stacking (linear meta-learner)
    stack_preds = stacking(
        scores_train_list, y_train, scores_val_list, reg=1.0,
    )
    results['stacking_linear'] = np.mean(stack_preds == y_val.astype(int))

    # 4b. Stacking (poly meta-learner)
    stack_preds_poly = stacking(
        scores_train_list, y_train, scores_val_list,
        reg=1.0, kernel_func=polynomial_kernel, degree=2, coef0=1.0,
    )
    results['stacking_poly'] = np.mean(stack_preds_poly == y_val.astype(int))

    # ── Print results ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {'Strategy':<30s} {'Val Acc':>8s}")
    print(f"  {'-'*40}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = ' *' if acc == max(results.values()) else ''
        print(f"  {name:<30s} {acc:>8.4f}{marker}")
    print(f"{'='*55}")

    return results


# ── Main: end-to-end demo ────────────────────────────────────────────

if __name__ == '__main__':
    import time
    from data import load_data, split_data_train_val
    from CKN.network import CKNNetwork
    from CKN.preprocessing import PreprocessV2

    # ── Load data ─────────────────────────────────────────────────
    X_train_full, X_test, y_train_full = load_data('data')
    X_train_flat, X_val_flat, y_train, y_val = split_data_train_val(
        X_train_full, y_train_full, val_size=0.2, seed=42,
    )
    print(f"Train: {X_train_flat.shape[0]}, Val: {X_val_flat.shape[0]}")

    # ── Preprocess (ZCA) ──────────────────────────────────────────
    prep = PreprocessV2(eps=0.1)
    prep.fit(X_train_flat)
    train_imgs = prep.transform(X_train_flat)
    val_imgs = prep.transform(X_val_flat)

    # ── CKN feature extraction (small for demo) ──────────────────
    ckn_config = [
        {'patch_size': 3, 'n_filters': 64, 'subsampling': 2, 'sigma': None},
        {'patch_size': 3, 'n_filters': 64, 'subsampling': 2, 'sigma': None},
    ]
    print("\nTraining CKN (n_filters=64, 2 layers)...")
    ckn = CKNNetwork(ckn_config)
    ckn.unsup_train_all(train_imgs, max_patches=50000, n_pairs=3000)

    F_train = ckn.extract_features(train_imgs)
    F_val = ckn.extract_features(val_imgs)

    # Center + L2-normalize so polynomial kernel stays bounded
    mu = F_train.mean(0)
    F_train = F_train - mu
    F_val = F_val - mu
    tr_norms = np.linalg.norm(F_train, axis=1, keepdims=True)
    tr_norms[tr_norms < 1e-8] = 1.0
    F_train = F_train / tr_norms
    val_norms = np.linalg.norm(F_val, axis=1, keepdims=True)
    val_norms[val_norms < 1e-8] = 1.0
    F_val = F_val / val_norms
    print(f"Feature dim: {F_train.shape[1]}")

    # ── Run ensemble comparison ───────────────────────────────────
    t0 = time.time()
    results = run_ensemble_comparison(F_train, y_train, F_val, y_val)
    print(f"\nEnsemble comparison took {time.time() - t0:.1f}s")
