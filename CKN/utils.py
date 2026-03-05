import numpy as np
import scipy.ndimage

def extract_patches(image : np.ndarray, patch_size :int) : 
    """
    Extract patches from an image or characteristic.

    :param image: Tensor of size (C,H,W)
    :type image: np.ndarray
    :param patch_size: Size of the patch
    :type patch_size: int

    :return: Patch tensor flattened, (H_out * W_out, C * patch_size * patch_size) and size of outputs (H_out, W_out)    
    """
    C, H, W = image.shape
    h, w = patch_size, patch_size
    
    windows = np.lib.stride_tricks.sliding_window_view(image, (C, h, w), axis=(0, 1, 2))
    windows = windows[0]
    H_out, W_out = windows.shape[0], windows.shape[1]
    patches_flat = windows.reshape(H_out * W_out, -1)
    
    return patches_flat, (H_out, W_out)

def normalize_patches(patches, epsilon=1e-6):
    """
    Normalize the patches with a L2 norm. 
    CKN need normalized patches to be good

    :param patches: patchs matrix (N, patch_dim)
    :param epsilon: small values to avoid 0 division.
        
    :return: Normalized patches and originales norms
    """
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    norms_clipped = np.maximum(norms, epsilon)
    normalized_patches = patches / norms_clipped
    
    return normalized_patches, norms

def spherical_kmeans(patches, n_filters, max_iters=50):
    """
    Apprend les filtres (points d'ancrage) de manière non supervisée.
    Cette fonction projette les centres sur la sphère unité à chaque étape.

    Initialize spherical kmeans to initialize W. 
    """


    indices = np.random.choice(patches.shape[0], n_filters, replace=False)
    centroids = patches[indices].copy()
    centroids, _ = normalize_patches(centroids)
    
    for _ in range(max_iters):
        similarities = np.dot(patches, centroids.T)
        assignments = np.argmax(similarities, axis=1)
        new_centroids = np.zeros_like(centroids)
        for k in range(n_filters):
            cluster_points = patches[assignments == k]
            if len(cluster_points) > 0:
                new_centroids[k] = np.sum(cluster_points, axis=0)
            else:
                new_centroids[k] = centroids[k] 
        centroids, _ = normalize_patches(new_centroids)
        
    return centroids

def ckn_activation(normalized_patches, norms, filters, eta, sigma):
    """
    Apply CKN convolution and activation function. Algo 2 of the paper.

    :param normalized_patches: normalized entry patches
    :param norms: Original norms of patches
    :param filters: Learned filters W.
    :param eta: Importance weights
    :param sigma: Smooth the gaussian kernel 
    """

    dot_products = np.dot(normalized_patches, filters.T)
    distances_sq = 2.0 - 2.0 * dot_products
    
    kernel_vals = np.exp(-distances_sq / (2 * sigma ** 2))
    
    weighted_kernel_vals = np.sqrt(eta) * kernel_vals
    
    activations = norms * weighted_kernel_vals
    
    return activations

def gaussian_pooling(feature_map, subsampling_factor, beta):
    C, H, W = feature_map.shape
    pooled_map = []
    
    for c in range(C):
        blurred = scipy.ndimage.gaussian_filter(feature_map[c], sigma=beta)
        subsampled = blurred[::subsampling_factor, ::subsampling_factor]
        pooled_map.append(subsampled)
    
    return np.array(pooled_map)

import numpy as np
import scipy.optimize

def optimize_W_and_eta(patches, n_filters, sigma, n_pairs=1000):
    """
    Algo 1 optimization
    """
    n_samples = patches.shape[0]
    patch_dim = patches.shape[1]
    
    idx_x = np.random.choice(n_samples, n_pairs, replace=False)
    idx_y = np.random.choice(n_samples, n_pairs, replace=False)
    X = patches[idx_x]
    Y = patches[idx_y]
    
    if sigma is None:
        distances = np.linalg.norm(X - Y, axis=1)
        sigma = np.quantile(distances, 0.1)

    dist_sq_xy = 2.0 - 2.0 * np.sum(X * Y, axis=1) 
    K_xy = np.exp(-dist_sq_xy / (2 * (sigma ** 2))) 
    
    W_init = spherical_kmeans(patches, n_filters, max_iters=10)
    eta_init = np.ones(n_filters)
    
    params_init = np.concatenate([W_init.flatten(), eta_init])
    
    def loss_fn(params):
        W_flat = params[:-n_filters]
        eta = params[-n_filters:]
        W = W_flat.reshape(n_filters, patch_dim)
        
        dist_sq_xw = 2.0 - 2.0 * np.dot(X, W.T)
        K_xw = np.exp(-dist_sq_xw / (2 *sigma ** 2))
        
        dist_sq_yw = 2.0 - 2.0 * np.dot(Y, W.T)
        K_yw = np.exp(-dist_sq_yw / (2 *sigma ** 2))
        
        prediction = np.sum(eta * K_xw * K_yw, axis=1)
        
        error = K_xy - prediction
        return np.mean(error ** 2)

    bounds = [(None, None)] * (n_filters * patch_dim) + [(0, None)] * n_filters
    
    result = scipy.optimize.minimize(
        loss_fn, 
        params_init, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'disp': True, 'maxiter': 50} 
    )
    
    W_opt_flat = result.x[:-n_filters]
    eta_opt = result.x[-n_filters:]
    W_opt = W_opt_flat.reshape(n_filters, patch_dim)
    
    W_opt, _ = normalize_patches(W_opt)
    
    print(f"Final loss : {result.fun:.6f}")
    return W_opt, eta_opt, sigma



# # ==========================================
# # TEST DES FONCTIONS (Pour vérifier que tout marche)
# # ==========================================
# if __name__ == "__main__":
#     print("--- Test des briques de base du CKN ---")
    
#     # 1. Création d'une image fictive (3 canaux RGB, 32x32)
#     image = np.random.randn(3, 32, 32)
#     print(f"Image originale : {image.shape}")
    
#     # 2. Extraction des patchs 3x3
#     patch_size = 3
#     patches, (H_out, W_out) = extract_patches(image, patch_size)
#     print(f"Patchs extraits : {patches.shape} (H_out={H_out}, W_out={W_out})")
    
#     # 3. Normalisation
#     norm_patches, norms = normalize_patches(patches)
#     print(f"Patchs normalisés : {norm_patches.shape}, Normes : {norms.shape}")
    
#     # 4. Apprentissage (ALGORITHME 1 COMPLET)
#     n_filters = 16
#     sigma = 0.5
#     # On utilise 500 paires pour que le test tourne vite dans le terminal
#     W, eta = optimize_W_and_eta(norm_patches, n_filters, sigma, n_pairs=500)
#     print(f"Filtres appris (W) : {W.shape}")
#     print(f"Poids appris (eta) : {eta.shape}")
    
#     # 5. Activation CKN (ALGORITHME 2 - Étape 1)
#     # On passe bien "eta" en paramètre maintenant !
#     activations_flat = ckn_activation(norm_patches, norms, W, eta, sigma)
#     # On reforme l'image spatiale : (Canaux_out, H_out, W_out)
#     activations = activations_flat.reshape(H_out, W_out, n_filters).transpose(2, 0, 1)
#     print(f"Carte d'activation : {activations.shape}")
    
#     # 6. Pooling Gaussien (ALGORITHME 2 - Étapes 2 & 3)
#     subsampling = 2
#     beta = 1.0
#     pooled_output = gaussian_pooling(activations, subsampling, beta)
#     print(f"Sortie après pooling : {pooled_output.shape}")