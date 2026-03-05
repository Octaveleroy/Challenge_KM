from CKN.layer import CKNLayer
import numpy as np

class CKNNetwork:
    """
    Multi-layer CKN network
    """
    def __init__(self, layers_config):
        """
        layers_config : liste de dicts, ex:
        [
            {'patch_size': 3, 'n_filters': 32, 'subsampling': 2, 'sigma': 0.6},
            {'patch_size': 3, 'n_filters': 64, 'subsampling': 2, 'sigma': 0.6},
        ]
        """
        self.layers = [CKNLayer(**cfg) for cfg in layers_config]

    def unsup_train_all(self, images, max_patches=50000, n_pairs=2000):
        """
        Entraîne les couches une par une (greedy layer-wise)
        """
        current_maps = images  # liste d'images (C, H, W)

        for i, layer in enumerate(self.layers):
            print(f"Training layer {i+1}/{len(self.layers)}...")
            
            # 1. Entraîner la couche courante sur les feature maps actuelles
            layer.unsup_train(current_maps, max_patches=max_patches, n_pairs=n_pairs)
            
            # 2. Propager TOUTES les images pour obtenir l'entrée de la couche suivante
            current_maps = [layer.forward(img) for img in current_maps]
            print(f"  Output shape: {current_maps[0].shape}")

    def extract_features(self, images):
        """
        Forward pass complet → vecteur de features plat pour le SVM
        """
        features = []
        for img in images:
            x = img
            for layer in self.layers:
                x = layer.forward(x)
            features.append(x.flatten())
        return np.array(features)  # (N, feature_dim)