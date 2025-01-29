import numpy as np

def approach1_multi_hot_distance(seqA, seqB, weights=None):
    """
    Computes the Weighted Hamming distance for multi-hot binary data.

    Args:
        seqA (np.ndarray): A (T, B) array for the first sequence, where T is the time dimension, and B is the behavior dimension.
        seqB (np.ndarray): A (T, B) array for the second sequence, where T is the time dimension, and B is the behavior dimension.
        weights (np.ndarray or None): A 1D array of shape (B,) representing weights for each behavior.
                                       If None, uniform weights are used (weights = 1 for all behaviors).

    Returns:
        float: The Weighted Hamming distance between `seqA` and `seqB`.

    Raises:
        ValueError: If `seqA` and `seqB` have mismatched shapes.
    """
    # Validate input shapes
    if seqA.shape != seqB.shape:
        raise ValueError(f"Shape mismatch: seqA has shape {seqA.shape}, seqB has shape {seqB.shape}.")

    # Extract dimensions
    T, B = seqA.shape

    # Initialize weights if not provided
    if weights is None:
        weights = np.ones(B, dtype=float)
    elif len(weights) != B:
        raise ValueError(f"Weight shape mismatch: weights length is {len(weights)}, but expected {B}.")

    # Compute the element-wise inequality (Hamming difference)
    hamming_diff = seqA != seqB  # Shape (T, B), binary mask for differences

    # Weight the differences along the behavior dimension
    weighted_diff = hamming_diff * weights  # Broadcasting weights to shape (T, B)

    # Sum the weighted differences across time (T) and behaviors (B)
    dist_val = np.sum(weighted_diff)

    return dist_val
