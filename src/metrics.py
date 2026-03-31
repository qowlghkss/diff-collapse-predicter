import numpy as np
from scipy import linalg

def check_path_existence(alphas: np.ndarray, betas: np.ndarray, gamma: float = 1.0) -> float:
    """
    Calculates the Marginal Path Collapse probability (PEC).
    Reference: Path Existence Check for Diffusion Models.
    
    Args:
        alphas: Noise schedule coefficients (1 - beta).
        betas: Noise schedule coefficients.
        gamma: Scaling weight for the path existence calculation.
        
    Returns:
        Probability of path existence (inverse of collapse).
    """
    # Simplified PEC formula based on cumulative noise product
    alpha_cumprod = np.cumprod(alphas)
    
    # Path existence score is often related to the ratio of signal to noise
    # as the diffusion process progresses.
    signal_power = alpha_cumprod
    noise_power = 1.0 - alpha_cumprod
    
    # Marginal path existence probability
    pec_scores = signal_power / (noise_power + 1e-8)
    pec_prob = np.mean(np.exp(-gamma * pec_scores))
    
    return float(np.clip(1.0 - pec_prob, 0.0, 1.0))

def calculate_vendi_score(features: np.ndarray, kernel_type: str = "rbf", sigma: float = 1.0) -> float:
    """
    Calculates the Vendi Score for diversity measurement.
    V(K) = exp(H(lambda)), where H is Shannon entropy of eigenvalues.
    
    Args:
        features: Feature matrix of shape (N, D).
        kernel_type: Similarity kernel to use ('rbf' or 'linear').
        sigma: Kernel bandwidth for RBF.
        
    Returns:
        Vendi Score (diversity value).
    """
    n = features.shape[0]
    if n <= 1:
        return 1.0
        
    # Standardize features
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
    # Compute similarity matrix K
    if kernel_type == "linear":
        K = np.dot(features, features.T) / features.shape[1]
    else: # RBF
        sq_norms = np.sum(features**2, axis=1)
        dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(features, features.T)
        K = np.exp(-dist_sq / (2 * sigma**2))
    
    # Ensure K is PSD (Property of Vendi Score)
    K = K / n
    
    # Eigenvalues
    eigenvals = linalg.eigvalsh(K)
    eigenvals = np.maximum(eigenvals, 1e-12) # clip for log
    
    # Shannon Entropy
    entropy = -np.sum(eigenvals * np.log(eigenvals))
    
    # Vendi Score
    return float(np.exp(entropy))

def summarize_diversity(y_prob: np.ndarray, features: np.ndarray) -> dict:
    """Combines collapse probability and diversity score."""
    collapse_prob = float(np.mean(y_prob))
    diversity = calculate_vendi_score(features)
    return {
        "mean_collapse_prob": collapse_prob,
        "vendi_score": diversity
    }
