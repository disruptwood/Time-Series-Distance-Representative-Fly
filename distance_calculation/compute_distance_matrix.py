import numpy as np
from itertools import combinations
import concurrent.futures

# Import each approach's distance function
from distance_calculation.approach1_multi_hot import approach1_multi_hot_distance
from distance_calculation.approach2_intervals import approach2_interval_distance
from distance_calculation.approach3_markov import approach3_markov_distance
from distance_calculation.approach4_scores import approach4_scores_distance
from distance_calculation.approach5_simple_euclidean import approach5_simple_euclidean_distance


def compute_distance_matrix(all_flies, approach_id, parallel=True):
    """
    Builds an NxN distance matrix for the given list of flies (all_flies),
    using one of five possible approaches:
      1 => approach1_multi_hot_distance
      2 => approach2_interval_distance
      3 => approach3_markov_distance
      4 => approach4_scores_distance
      5 => approach5_simple_euclidean_distance

    Parameters:
      all_flies: list of data entries. For approaches:
         1 & 3 => each entry is a (T, B) multi-hot array
         2     => each entry is an interval dictionary
         4 & 5 => each entry is a (T, B) numeric array
      approach_id: integer 1..5 specifying which distance function to use.
      parallel: Boolean indicating whether to compute distances in parallel.

    Returns:
      An NxN numpy array containing pairwise distances.
    """

    # Dictionary dispatch for the distance functions
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

    # If parallel execution is enabled and there's more than one fly
    if parallel and N > 1:
        # Create all unique index pairs (i, j) for the upper triangle
        pairs = list(combinations(range(N), 2))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all pairwise distance computations
            future_to_pair = {
                executor.submit(dist_func, all_flies[i], all_flies[j]): (i, j)
                for i, j in pairs
            }
            # As each future completes, fill in both symmetric entries
            for future in concurrent.futures.as_completed(future_to_pair):
                i, j = future_to_pair[future]
                dist_val = future.result()
                dist_mat[i, j] = dist_val
                dist_mat[j, i] = dist_val
    else:
        # Sequential computation
        for i in range(N):
            for j in range(i + 1, N):
                dist_val = dist_func(all_flies[i], all_flies[j])
                dist_mat[i, j] = dist_val
                dist_mat[j, i] = dist_val

    return dist_mat
