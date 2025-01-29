import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def approach4_scores_distance(seqA, seqB):
    """
    Computes the Dynamic Time Warping (DTW) distance between two T × B numeric score arrays.

    Args:
        seqA (np.ndarray): A (T, B) numeric score array for the first sequence.
        seqB (np.ndarray): A (T, B) numeric score array for the second sequence.

    Returns:
        float: DTW distance between `seqA` and `seqB`.

    Raises:
        ValueError: If the input sequences are not 2D arrays.
    """
    if seqA.ndim != 2 or seqB.ndim != 2:
        raise ValueError("Input sequences must be 2D arrays with shape (T, B).")

    # Use fastdtw to compute DTW distance with Euclidean base metric
    dist, _ = fastdtw(seqA, seqB, dist=euclidean)
    return dist
