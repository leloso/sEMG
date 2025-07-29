import torch
import torch.nn as nn


class RBF(nn.Module):
    """
    Radial Basis Function (RBF) kernel for Maximum Mean Discrepancy (MMD) computation.
    
    Uses multiple Gaussian kernels with different bandwidths to capture various scales
    of similarity between data points. The kernel computes:
    K(x, y) = exp(-||x - y||² / (2 * σ²))
    
    Args:
        n_kernels (int): Number of kernels with different bandwidths. Default: 5
        mul_factor (float): Multiplicative factor between consecutive bandwidths. Default: 2.0
        bandwidth (float, optional): Fixed bandwidth. If None, uses median heuristic. Default: None
    """
    
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        # Create bandwidth multipliers: [..., 1/4, 1/2, 1, 2, 4, ...] for n_kernels=5
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        """
        Compute bandwidth using median heuristic if not provided.
        
        The median heuristic sets bandwidth to the median of pairwise distances,
        approximated here as mean of all pairwise distances.
        
        Args:
            L2_distances (torch.Tensor): Pairwise L2 distances matrix
            
        Returns:
            float: Computed bandwidth
        """
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            # Median heuristic: use mean of pairwise distances as bandwidth estimate
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        """
        Compute RBF kernel matrix for input data.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Kernel matrix of shape (n_samples, n_samples)
                         Sum of multiple RBF kernels with different bandwidths
        """
        # Compute pairwise squared L2 distances: ||x_i - x_j||²
        L2_distances = torch.cdist(X, X) ** 2
        
        # Get bandwidth (median heuristic or fixed)
        bandwidth = self.get_bandwidth(L2_distances)
        
        # Compute RBF kernel for each bandwidth: exp(-d²/(bandwidth * multiplier))
        # Shape: (n_kernels, n_samples, n_samples)
        kernels = torch.exp(-L2_distances[None, ...] / 
                           (bandwidth * self.bandwidth_multipliers)[:, None, None])
        
        # Sum over all kernels to get final kernel matrix
        return kernels.sum(dim=0)


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss for measuring distributional differences.
    
    MMD is a kernel-based statistical test that measures the distance between
    two probability distributions by comparing their mean embeddings in a 
    reproducing kernel Hilbert space (RKHS).
    
    The MMD² between distributions P and Q is computed as:
    MMD²(P, Q) = E[k(X, X')] - 2*E[k(X, Y)] + E[k(Y, Y')]
    
    where X ~ P, Y ~ Q, and k is a characteristic kernel (e.g., RBF).
    
    This implementation uses the unbiased empirical estimator:
    MMD²(X, Y) = (1/n²)∑k(x_i, x_j) - (2/nm)∑k(x_i, y_j) + (1/m²)∑k(y_i, y_j)
    
    Args:
        kernel (nn.Module): Kernel function to use. Default: RBF kernel with 5 bandwidths
        
    Example:
        >>> mmd_loss = MMDLoss()
        >>> source_data = torch.randn(100, 64)  # Source distribution samples
        >>> target_data = torch.randn(120, 64)  # Target distribution samples
        >>> loss = mmd_loss(source_data, target_data)
        >>> print(f"MMD loss: {loss.item():.4f}")
    
    """
    
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        """
        Compute MMD loss between two sets of samples.
        
        Args:
            X (torch.Tensor): Source samples of shape (n_samples, n_features)
            Y (torch.Tensor): Target samples of shape (m_samples, n_features)
            
        Returns:
            torch.Tensor: Scalar MMD² loss value
            
        Note:
            - Higher values indicate more dissimilar distributions
            - Zero value indicates identical distributions (in the limit)
            - Always non-negative due to the kernel's positive definiteness
        """
        # Concatenate samples to compute joint kernel matrix efficiently
        # Combined shape: (n_samples + m_samples, n_features)
        combined = torch.vstack([X, Y])
        
        # Compute kernel matrix for all pairs: K(x_i, x_j), K(x_i, y_j), K(y_i, y_j)
        K = self.kernel(combined)
        
        X_size = X.shape[0]
        
        # Extract kernel submatrices
        # XX: kernel values between source samples K(x_i, x_j)
        XX = K[:X_size, :X_size].mean()
        
        # XY: kernel values between source and target samples K(x_i, y_j)  
        XY = K[:X_size, X_size:].mean()
        
        # YY: kernel values between target samples K(y_i, y_j)
        YY = K[X_size:, X_size:].mean()
        
        # Compute unbiased MMD² estimator
        # MMD²(X, Y) = E[K(X,X)] - 2*E[K(X,Y)] + E[K(Y,Y)]
        return XX - 2 * XY + YY


class CombinedLoss(nn.Module):
    """
    Combined loss function for domain adaptation training
    
    This combines the classification loss (e.g., CrossEntropy) with MMD loss
    for joint optimization during training.
    
    Args:
        classification_loss: Base classification loss (e.g., nn.CrossEntropyLoss())
        mmd_loss: MMD loss instance
        C: Weight for MMD loss term (lambda parameter)
    """
    
    def __init__(self, classification_loss, mmd_loss, C=1.0):
        super(CombinedLoss, self).__init__()
        self.classification_loss = classification_loss
        self.mmd_loss = mmd_loss
        self.C = C
        
    def forward(self, predictions, labels, source_features, target_features):
        """
        Compute combined loss
        
        Args:
            predictions: Model predictions for labeled source data
            labels: Ground truth labels for source data
            source_features: Feature representations from source domain
            target_features: Feature representations from target domain
            
        Returns:
            combined_loss: Weighted sum of classification and MMD losses
            loss_dict: Dictionary containing individual loss components
        """
        # Classification loss on labeled source data
        cls_loss = self.classification_loss(predictions, labels)
        
        # MMD loss between source and target features
        mmd_loss_val = self.mmd_loss(source_features, target_features)
        
        # Combined loss
        total_loss = cls_loss + self.C * mmd_loss_val
        
        loss_dict = {
            'classification_loss': cls_loss.item(),
            'mmd_loss': mmd_loss_val.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


# Example usage
if __name__ == "__main__":
    # Initialize losses
    mmd_loss = MMDLoss()
    classification_loss = nn.CrossEntropyLoss()
    combined_loss = CombinedLoss(classification_loss, mmd_loss, C=1)
    
    # Example data
    batch_size = 256
    feature_dim = 256
    num_classes = 7
    
    # Source domain data (labeled)
    source_features = torch.randn(batch_size, feature_dim)
    source_labels = torch.randint(0, num_classes, (batch_size,))
    source_predictions = torch.randn(batch_size, num_classes)
    
    # Target domain data (unlabeled)
    target_features = torch.randn(batch_size, feature_dim)
    
    # Compute combined loss
    total_loss, loss_components = combined_loss(
        source_predictions, source_labels, 
        source_features, target_features
    )
    
    print("Loss components:")
    for key, value in loss_components.items():
        print(f"{key}: {value:.4f}")