# Kernel Methods Data Challenge (2025-2026)

**Team Name:** Martin Leroy  
**Members:** Tristan Martin, Octave Leroy

---

## Overview

This repository contains our from-scratch implementation for the Kernel Methods data challenge. The objective is to classify 32×32 RGB images into 10 categories. All pipeline components are implemented using only basic linear algebra and optimization primitives.

Classical approaches like Kernel Ridge Regression (KRR) and Support Vector Machines (SVM) plateaued below 50% accuracy when applied to raw flattened pixels, because global kernels discard spatial locality and translation invariance. To solve this, we implemented an unsupervised Convolutional Kernel Network (CKN) to approximate a non-linear kernel over local image patches, producing spatially-structured, translation-invariant features. These features are then passed to a KRR classifier.

---

## Repository Structure

```text
C:.
│   augment.py          # Augment the data with flipping and shifting
│   data.py             # Load the data and split it
│   kernels.py          # Implement basic kernels
│   prepocess.py        # SIFT implementation
│   utils.py            # Accuracy and submission helpers
│   visualization.py    # Visualize images
│   challenge.ipynb     # Execute the full pipeline to obtain results
│
├───CKN                 # Convolutional Kernel Network implementation
│       layer.py        # Single CKN layer
│       network.py      # Full CKN network
│       utils.py        # Basic CKN components
│
└───classifier          # Classifiers
        KRR.py          # KRR implemented as a One-vs-Rest 
        SVM.py          # SVM implemented as a One-vs-Rest 
```

---

## Data Preprocessing

- **ZCA Whitening:** Empirical covariance inspection revealed the provided images were already ZCA whitened.
- **Standardization:** We applied channel-wise Z-score normalization.
- **Data Augmentation:** Horizontal flipping and random translations of up to 2 pixels were used to mitigate overfitting.

---

## Results

Our approach significantly improved validation accuracy on the dataset compared to baseline methods:

| Method    | Features                    | Accuracy (%) |
|-----------|-----------------------------|--------------|
| KRR       | Polynomial, raw pixels      | 42.0         |
| SVM       | Polynomial, SIFT            |  32.60       |
| CKN + KRR | p1=256, 3×3 patches         | 64.0         |

---

## Optimal Configuration & Usage

Our best-performing model uses a single-layer CKN followed by a KRR classifier.

**Optimal Hyperparameters:**

- **CKN:** `patch_size` = 3, `n_filters` = 256, `subsampling` = 2
- **KRR:** Polynomial kernel, `reg` = 0.8, `degree` = 4, `coef0` = 6

This configuration is already implemented in the start.py file. However it takes around 45 min to complete


## Quick Start

**1. Download the data**

Download the challenge dataset from the platform and place the three files in the `./data/` directory:

```text
data/
├── Xtr.csv
├── Xte.csv
└── Ytr.csv
```

**2. Run the pipeline**

```bash
python start.py
```

This script executes the full pipeline end-to-end : data loading, augmentation, CKN training, feature extraction, KRR classification. It also saves the submission file in `./submissions/`. No additional configuration is required.

---