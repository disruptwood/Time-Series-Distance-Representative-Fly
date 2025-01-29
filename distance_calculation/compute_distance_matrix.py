import numpy as np

# Import each approach's distance function
from distance_calculation.approach1_multi_hot import approach1_multi_hot_distance
from distance_calculation.approach2_intervals import approach2_interval_distance
from distance_calculation.approach3_markov import approach3_markov_distance
from distance_calculation.approach4_scores import approach4_scores_distance
from distance_calculation.approach5_simple_euclidean import approach5_simple_euclidean_distance


def compute_distance_matrix(all_flies, approach_id):
    """
    Builds an NxN distance matrix for the given list of flies (all_flies),
    using one of five possible approaches:
      1 => approach1_multi_hot_distance
      2 => approach2_interval_distance
      3 => approach3_markov_distance
      4 => approach4_scores_distance
      5 => approach5_simple_euclidean_distance

    all_flies: list of data entries. For approaches:
      1 & 3 => each entry is a (T, B) multi-hot array
      2     => each entry is an interval dictionary
      4 & 5 => each entry is a (T, B) numeric array
    approach_id: integer 1..5

    Returns an NxN numpy array dist_mat.
    """

    # Dictionary dispatch for cleaner code
    distance_funcs = {
        1: approach1_multi_hot_distance,
        2: approach2_interval_distance,
        3: approach3_markov_distance,
        4: approach4_scores_distance,
        5: approach5_simple_euclidean_distance
    }

    if approach_id not in distance_funcs:
        raise ValueError(f"Invalid approach_id: {approach_id}")

    dist_func = distance_funcs[approach_id]

    N = len(all_flies)
    dist_mat = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(i+1, N):
            dist_val = dist_func(all_flies[i], all_flies[j])
            dist_mat[i, j] = dist_val
            dist_mat[j, i] = dist_val

    return dist_mat
