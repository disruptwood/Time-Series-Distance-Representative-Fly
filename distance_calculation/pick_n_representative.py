import numpy as np

def pick_n_representatives_simple(dist_mat, index_map, n):
    """
    Selects the top n most representative flies from dist_mat.
    Representation is determined by summing each fly's total distance
    to all other flies, then picking the smallest sums.

    dist_mat: NxN numpy array of pairwise distances.
    index_map: list of (condition_idx, local_fly_idx) for each row i in dist_mat.
    n: integer, how many representative flies to pick.

    Returns a list of (global_idx, sum_dist). The lowest sum_dist is most representative.
    """

    N = dist_mat.shape[0]
    if n > N:
        raise ValueError(f"Requested {n} representatives, but only have {N} flies.")

    # Sum distances for each fly
    dist_sums = np.sum(dist_mat, axis=1)  # shape (N,)

    # Sort by ascending sum
    ranking = sorted(enumerate(dist_sums), key=lambda x: x[1])  # [(i, sum), ...]
    # Take the top n
    chosen = ranking[:n]  # each item is (global_idx, sum_dist)

    return chosen
