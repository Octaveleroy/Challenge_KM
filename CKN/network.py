from CKN.layer import CKNLayer
import numpy as np

class CKNNetwork:
    """
    Multi-layer CKN network
    """
    def __init__(self, layers_config):
        self.layers = [CKNLayer(**cfg) for cfg in layers_config]

    def unsup_train_all(self, images, max_patches=50000, n_pairs=2000):
        """
        Train layer per layer
        """
        current_maps = images  # (C, H, W)

        for i, layer in enumerate(self.layers):
            print(f"Training layer {i+1}/{len(self.layers)}...")
            
            # 1. Train the current layer on the actual feature maps
            layer.unsup_train(current_maps, max_patches=max_patches, n_pairs=n_pairs)
            
            # 2. Propagate to obtain the entry to the next layer
            current_maps = [layer.forward(img) for img in current_maps]
            print(f"  Output shape: {current_maps[0].shape}")

    def extract_features(self, images):
        """
        Complete forward pass and flattent for Classifier
        """
        features = []
        for img in images:
            x = img
            for layer in self.layers:
                x = layer.forward(x)
            features.append(x.flatten())
        return np.array(features)  # (N, feature_dim)