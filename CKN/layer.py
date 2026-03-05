import numpy as np
from CKN.utils import (
    extract_patches, normalize_patches, 
    spherical_kmeans, ckn_activation, gaussian_pooling,
    optimize_W_and_eta
)

class CKNLayer:
    """
    CKN Layer that have full train logics and transformation
    """
    def __init__(self, patch_size, n_filters, subsampling,sigma = None):
        self.patch_size = patch_size
        self.n_filters = n_filters
        self.subsampling = subsampling
        self.sigma = sigma
        self.beta = float(subsampling)
        
        self.W = None 
        self.eta = None

    def unsup_train(self, input_maps, max_patches=100000, n_pairs=5000):
        """
        ALGO 1, training filters
        """
        all_patches = []
        
        for img in input_maps:
            patches, _ = extract_patches(img, self.patch_size)
            all_patches.append(patches)
            if sum(len(p) for p in all_patches) > max_patches:
                break
                
        X = np.vstack(all_patches)[:max_patches]
      
        X_norm, _ = normalize_patches(X)

        print(f"  Patches norm mean: {np.linalg.norm(X_norm, axis=1).mean():.3f}")  # doit être ~1.0
        print(f"  Patches raw range: [{X.min():.3f}, {X.max():.3f}]")
        
        self.W, self.eta, self.sigma = optimize_W_and_eta(X_norm, self.n_filters, self.sigma, n_pairs=n_pairs)

    def forward(self, image):
        """
        Algo 2, going to other layer
        """
        if self.W is None or self.eta is None:
            raise ValueError("The first layer doesn't be trained")
            
        patches, (H_out, W_out) = extract_patches(image, self.patch_size)
        norm_patches, norms = normalize_patches(patches)
        
        activations_flat = ckn_activation(norm_patches, norms, self.W, self.eta, self.sigma)
        activations = activations_flat.reshape(H_out, W_out, self.n_filters).transpose(2, 0, 1)
        output = gaussian_pooling(activations, self.subsampling, self.beta)
        
        return output