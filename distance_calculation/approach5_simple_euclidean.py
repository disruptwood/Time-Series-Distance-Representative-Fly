import numpy as np

def approach5_simple_euclidean_distance(seqA, seqB):
    """
    Computes the Euclidean distance between two binary postprocessed data arrays.

    Args:
        seqA (np.ndarray): A (T, B) binary array for the first sequence, where T is the time dimension, and B is the behavior dimension.
        seqB (np.ndarray): A (T, B) binary array for the second sequence, where T is the time dimension, and B is the behavior dimension.

    Returns:
        float: Euclidean distance between `seqA` and `seqB`.

    Raises:
        ValueError: If the input sequences do not have the same shape.
    """
    # Validate input shapes
    if seqA.shape != seqB.shape:
        raise ValueError(f"Shape mismatch: seqA {seqA.shape}, seqB {seqB.shape}.")

    # Compute Euclidean distance
    distance = np.sqrt(np.sum((seqA - seqB) ** 2))
    return distance
