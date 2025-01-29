import os
import numpy as np
from scipy.io import loadmat

def load_scores_data(mat_files, single=True, index_map=None, condition_names=None):
    """
    Loads continuous "scores" data. Each .mat file is one behavior dimension,
    and we use scoreNorm to normalize each fly's data. If single=True,
    mat_files is a flat list. If single=False, mat_files is a list-of-lists.

    Returns:
      all_flies: list of (T, B) arrays (global indexing).
      index_map: global_index -> (cond_idx, local_fly_idx)
      condition_names: updated list of condition names
    """
    if single:
        # single condition => just load them as one group
        flies_cond, cond_name = _load_single_condition_scores(mat_files)
        if condition_names is None:
            condition_names = [cond_name]
        # build a global index_map
        index_map = {}
        for i in range(len(flies_cond)):
            index_map[i] = (0, i)
        return flies_cond, index_map, condition_names

    else:
        # multiple conditions => a list of lists
        if not mat_files or not isinstance(mat_files[0], list):
            raise RuntimeError("For multiple conditions, mat_files must be a list of lists.")

        if condition_names is None:
            condition_names = [f"Condition_{i}" for i in range(len(mat_files))]

        all_flies_global = []
        index_map_global = {}
        global_idx = 0

        for cond_idx, file_list in enumerate(mat_files):
            flies_cond, guess_name = _load_single_condition_scores(file_list)
            # If we have a user-specified name, override
            if cond_idx < len(condition_names):
                guess_name = condition_names[cond_idx]
                condition_names[cond_idx] = guess_name

            for local_fly_idx, arr in enumerate(flies_cond):
                all_flies_global.append(arr)
                index_map_global[global_idx] = (cond_idx, local_fly_idx)
                global_idx += 1

        return all_flies_global, index_map_global, condition_names


def _load_single_condition_scores(file_list):
    """
    Helper: loads a single condition's worth of .mat files, each is one behavior dimension.
    Normalizes each fly's data by that file's scoreNorm.

    Returns:
      all_flies_cond: a list of (T, B) arrays, one for each fly.
      cond_name: inferred from the directory of the first file (can be overridden).
    """
    if not file_list:
        return [], "SingleCondition"

    # Determine # of flies from the first .mat
    first_data = loadmat(file_list[0], struct_as_record=False, squeeze_me=True)['allScores']
    if not hasattr(first_data, 'scores') or not hasattr(first_data, 'scoreNorm'):
        raise RuntimeError("First file missing 'scores' or 'scoreNorm'.")

    # #flies from the first file
    num_flies = len(first_data.scores)
    # condition name guess
    cond_folder = os.path.dirname(os.path.abspath(file_list[0]))
    cond_name = os.path.basename(cond_folder)

    # We'll build for each fly => list of columns
    fly_behaviors = [[] for _ in range(num_flies)]
    max_lengths = [0]*num_flies

    for mf in file_list:
        mat_data = loadmat(mf, struct_as_record=False, squeeze_me=True)['allScores']
        if not hasattr(mat_data, 'scores') or not hasattr(mat_data, 'tStart') \
           or not hasattr(mat_data, 'tEnd') or not hasattr(mat_data, 'scoreNorm'):
            raise RuntimeError(f"Missing fields in {mf} (scores, tStart, tEnd, scoreNorm).")

        scores = mat_data.scores  # cell array, shape (#flies,)
        score_norm = float(mat_data.scoreNorm)

        t_starts = mat_data.tStart.flatten()
        t_ends   = mat_data.tEnd.flatten()

        if len(scores) != num_flies:
            raise RuntimeError(f"Inconsistent #flies in {mf}, expected {num_flies}, got {len(scores)}")

        for fly_i in range(num_flies):
            raw_arr = np.array(scores[fly_i]).ravel()  # shape ~ (video_len,)
            # normalize
            raw_arr = raw_arr / score_norm

            start_idx = int(t_starts[fly_i]) - 1
            end_idx   = int(t_ends[fly_i])
            # clamp to safe ranges in case of small off-by-one
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(raw_arr):
                end_idx = len(raw_arr)

            slice_i = raw_arr[start_idx:end_idx]

            # Accumulate
            fly_behaviors[fly_i].append(slice_i)
            if len(slice_i) > max_lengths[fly_i]:
                max_lengths[fly_i] = len(slice_i)

    # Now unify each fly => (T, B)
    B = len(file_list)
    all_flies_cond = []
    for i in range(num_flies):
        T = max_lengths[i]
        mat_ = np.full((T, B), np.nan, dtype=float)
        for b_idx, col_slice in enumerate(fly_behaviors[i]):
            mat_[:len(col_slice), b_idx] = col_slice
        all_flies_cond.append(mat_)

    return all_flies_cond, cond_name