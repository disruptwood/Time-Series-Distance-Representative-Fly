import os
import numpy as np
from scipy.io import loadmat


def load_postprocessed_data(mat_files, single=True, index_map=None, condition_names=None):
    """
    Loads postprocessed (binary) data. Works for:
      - single=True, where 'mat_files' is a flat list of .mat paths (one condition).
      - single=False, where 'mat_files' is a list of lists of .mat paths
        (each sub-list is a condition).

    Returns:
      all_flies: list of (T, B) arrays, one per fly (global indexing).
      index_map: global_index -> (condition_idx, local_fly_idx)
      condition_names: final list of condition names
    """
    # If single => dif logic
    if single:
        return _load_single_condition_postprocessed(mat_files, condition_names, index_map)
    else:
        if not mat_files:
            raise RuntimeError("Empty mat_files list for multiple conditions.")
        # Expect mat_files to be a list-of-lists. Each sublist => one condition
        if not isinstance(mat_files[0], list):
            raise RuntimeError("For multiple conditions, mat_files must be list of lists.")

        # If the caller didn't pass condition_names, build them
        if condition_names is None:
            condition_names = [f"Condition_{i}" for i in range(len(mat_files))]

        all_flies = []
        final_index_map = {}
        global_idx = 0

        for cond_idx, file_list in enumerate(mat_files):
            # Load this condition's flies
            flies_this_cond = _load_single_condition_postprocessed(
                file_list, single_condition_name=condition_names[cond_idx]
            )  # returns (flies, some dummy index_map, cond_names)
            # The function returns (all_flies_condition, local_index_map, [cond_name])
            flies_cond, local_map, _ = flies_this_cond

            # Now we incorporate these into the global list
            for local_fly_idx, fly_data in enumerate(flies_cond):
                all_flies.append(fly_data)
                final_index_map[global_idx] = (cond_idx, local_fly_idx)
                global_idx += 1

        return all_flies, final_index_map, condition_names


def _load_single_condition_postprocessed(file_list, single_condition_name=None, index_map=None):
    """
    Helper function that loads postprocessed data from a *single condition*
    given a list of .mat files (behaviors). Returns (list_of_flies, local_index_map, [cond_name]).
    Each fly => shape (T, B).
    """
    if not file_list:
        return [], {}, [single_condition_name or "SingleCondition"]

    first_data = loadmat(file_list[0])['allScores']
    if 'postprocessed' not in first_data.dtype.names:
        raise RuntimeError("First file missing 'postprocessed' field.")

    # Determine how many flies from the first file
    pp0 = first_data['postprocessed'][0, 0]
    num_flies = max(pp0.shape)

    fly_behaviors = [[] for _ in range(num_flies)]
    max_lengths = [0] * num_flies

    # We interpret each file in 'file_list' as a single "behavior" dimension
    for mf in file_list:
        mat_data = loadmat(mf)['allScores']
        if 'postprocessed' not in mat_data.dtype.names:
            raise RuntimeError(f"No 'postprocessed' in {mf}.")
        postprocessed = mat_data['postprocessed'][0, 0]
        t_starts = mat_data['tStart'][0, 0][0]
        t_ends = mat_data['tEnd'][0, 0][0]

        # Decide if shape is (#flies, something) or (something, #flies)
        rows, cols = postprocessed.shape
        if rows == num_flies:
            get_fly_pp = lambda i: postprocessed[i, 0]
        elif cols == num_flies:
            get_fly_pp = lambda i: postprocessed[0, i]
        else:
            raise RuntimeError(f"Mismatch #flies vs shape in {mf}.")

        for fly_i in range(num_flies):
            arr = get_fly_pp(fly_i).ravel()
            start_idx = max(0, t_starts[fly_i] - 1)
            end_idx = min(t_ends[fly_i], len(arr))

            valid_slice = arr[start_idx:end_idx]

            # Trim or pad if the length is off by a small margin
            # e.g. if end_idx is up to 3 frames beyond arr or if arr is short
            if end_idx - start_idx != len(valid_slice):
                # We'll rely on the min(...) usage above to avoid out-of-bounds
                pass

            # If you want to warn if final length < 27000
            if len(valid_slice) < 27000:
                # For example:
                print(
                    f"Warning: Condition={single_condition_name}, Fly={fly_i} slice length < 27000 => {len(valid_slice)}")

            fly_behaviors[fly_i].append(valid_slice)
            if len(valid_slice) > max_lengths[fly_i]:
                max_lengths[fly_i] = len(valid_slice)

    # Build each fly's (T, B)
    all_flies_condition = []
    B = len(file_list)
    for i in range(num_flies):
        T = max_lengths[i]
        mat = np.full((T, B), np.nan, dtype=float)
        for b_idx, col in enumerate(fly_behaviors[i]):
            mat[:len(col), b_idx] = col
        all_flies_condition.append(mat)

    # local index_map for this condition
    local_index_map = {fly_i: fly_i for fly_i in range(num_flies)}

    return all_flies_condition, local_index_map, [single_condition_name or "SingleCondition"]