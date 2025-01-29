import numpy as np
from scipy.io import loadmat

def load_intervals_data(mat_files, single=True, index_map=None, condition_names=None):
    """
    Loads t0s–t1s interval data. If single=True, 'mat_files' is a flat list for one condition.
    If single=False, 'mat_files' is a list-of-lists: each sub-list is one condition.

    Returns:
      all_flies: a global list of intervals, one entry per "fly" in total.
        (If you have multiple conditions each with N flies, you'll have M*N total entries.)
      index_map: global_index -> (condition_idx, local_fly_idx)
      condition_names: updated list of condition names
    """
    if single:
        # Single condition => old approach or call a helper
        all_flies, _, cond_name = _load_single_condition_intervals(mat_files)
        if condition_names is None:
            condition_names = [cond_name]

        # Build index_map
        index_map = {}
        for i in range(len(all_flies)):
            index_map[i] = (0, i)

        return all_flies, index_map, condition_names

    else:
        # Multiple conditions => mat_files is list of lists
        if not mat_files or not isinstance(mat_files[0], list):
            raise RuntimeError("For multiple conditions, mat_files must be a list of lists.")

        # If no condition_names provided, make some
        if condition_names is None:
            condition_names = [f"Condition_{i}" for i in range(len(mat_files))]

        all_flies_global = []
        index_map_global = {}
        global_idx = 0

        for cond_idx, file_list in enumerate(mat_files):
            # Load intervals for a single condition
            per_cond_flies, num_flies_cond, cond_name = _load_single_condition_intervals(file_list)
            # Overwrite the default cond_name with the provided name if we have one
            if cond_idx < len(condition_names):
                cond_name = condition_names[cond_idx]
                condition_names[cond_idx] = cond_name

            # Now unify them into the global list
            for local_fly_idx, intervals_for_this_fly in enumerate(per_cond_flies):
                all_flies_global.append(intervals_for_this_fly)
                index_map_global[global_idx] = (cond_idx, local_fly_idx)
                global_idx += 1

        return all_flies_global, index_map_global, condition_names


def _load_single_condition_intervals(file_list):
    """
    Helper for loading interval data for a *single* condition: a list of .mat files.
    We accumulate intervals across all files for each fly.

    Returns:
      all_flies_cond: list of length num_flies, each item is a list of intervals
                      (e.g., [{"tStart": X, "tEnd": Y}, ...]).
      num_flies: number of flies for this condition
      cond_name: a default single-condition name (can be overridden)
    """
    if not file_list:
        return [], 0, "SingleCondition"

    # Determine how many flies from the first file
    first_data = loadmat(file_list[0], struct_as_record=False, squeeze_me=True)['allScores']
    if not hasattr(first_data, 't0s') or not hasattr(first_data, 't1s'):
        raise RuntimeError("First file missing 't0s' or 't1s' fields.")

    num_flies = len(first_data.t0s)  # each entry is a list/array of starts
    all_flies_cond = [ [] for _ in range(num_flies) ]  # each item => list of intervals

    # We'll guess a cond_name from the folder of the first .mat
    import os
    cond_name = "SingleCondition"
    folder_of_first = os.path.dirname(os.path.abspath(file_list[0]))
    cond_name = os.path.basename(folder_of_first)

    for mf in file_list:
        mat_data = loadmat(mf, struct_as_record=False, squeeze_me=True)['allScores']
        if not hasattr(mat_data, 't0s') or not hasattr(mat_data, 't1s'):
            raise RuntimeError(f"File {mf} is missing 't0s' or 't1s'.")

        t0s = mat_data.t0s
        t1s = mat_data.t1s

        # Check consistency
        if len(t0s) != num_flies:
            raise ValueError(f"Inconsistent fly count in {mf} (expected {num_flies}, got {len(t0s)}).")
        if len(t1s) != num_flies:
            raise ValueError(f"Inconsistent fly count in {mf} (expected {num_flies}, got {len(t1s)}).")

        for fly_i in range(num_flies):
            fly_t0s = t0s[fly_i]
            fly_t1s = t1s[fly_i]
            # Could be scalar or array => make sure they are arrays
            if np.isscalar(fly_t0s):
                fly_t0s = np.array([fly_t0s])
            else:
                fly_t0s = np.array(fly_t0s).ravel()

            if np.isscalar(fly_t1s):
                fly_t1s = np.array([fly_t1s])
            else:
                fly_t1s = np.array(fly_t1s).ravel()

            if len(fly_t0s) != len(fly_t1s):
                raise ValueError(f"Fly {fly_i} in {mf} has mismatched t0s/t1s lengths.")

            intervals = []
            for start, end in zip(fly_t0s, fly_t1s):
                intervals.append({"tStart": int(start), "tEnd": int(end)})

            # Accumulate these intervals
            all_flies_cond[fly_i].extend(intervals)

    return all_flies_cond, num_flies, cond_name
