# main.py (root of project)

import sys
import numpy as np

# Interfaces
from interface.input_mode import pick_input_mode
from interface.approach_selection import pick_approach
from interface.choose_n_representatives import pick_n_representatives
from interface.pick_mat_files import pick_mat_files_single_condition
from interface.pick_folders import pick_multiple_folders_for_conditions

# Data loading
from data_loading.gather_mat_files_single import gather_mat_files_single_condition
from data_loading.gather_mat_files_multiple import gather_mat_files_multiple_condition
from data_loading.load_postprocessed import load_postprocessed_data
from data_loading.load_intervals import load_intervals_data
from data_loading.load_scores import load_scores_data
from data_loading.unify_lengths import unify_lengths

# Distance calculation
from distance_calculation.compute_distance_matrix import compute_distance_matrix
from distance_calculation.pick_n_representative import pick_n_representatives_simple


def main():
    # 1) Choose single or multiple
    mode = pick_input_mode()

    # 2) Choose approach
    approach_id = pick_approach()

    # 3) Choose number of representatives
    n = pick_n_representatives()

    # 4) Collect data
    if mode == 'single':
        # Single condition => pick .mat files
        mat_files = pick_mat_files_single_condition()
        # Gather them into a consistent list (if needed)
        flies_data_paths = gather_mat_files_single_condition(mat_files)

        # Depending on approach, load data appropriately
        if approach_id == 1:
            # e.g. load postprocessed binary
            all_flies, index_map, condition_names = load_postprocessed_data(flies_data_paths, single=True)
        elif approach_id == 2:
            # e.g. load intervals
            all_flies, index_map, condition_names = load_intervals_data(flies_data_paths, single=True)
        elif approach_id in [3, 4, 5]:
            # e.g. load scores or postprocessed again
            all_flies, index_map, condition_names = load_scores_data(flies_data_paths, single=True)
        else:
            print("Unknown approach.")
            sys.exit(1)

    else:
        # Multiple conditions => pick multiple folders in one dialog
        folder_list = pick_multiple_folders_for_conditions()
        # Gather them
        flies_data_paths, index_map, condition_names = gather_mat_files_multiple_condition(folder_list)

        # Then load based on approach
        if approach_id == 1:
            all_flies, index_map, condition_names = load_postprocessed_data(flies_data_paths, single=False,
                                                                            index_map=index_map,
                                                                            condition_names=condition_names)
        elif approach_id == 2:
            all_flies, index_map, condition_names = load_intervals_data(flies_data_paths, single=False,
                                                                        index_map=index_map,
                                                                        condition_names=condition_names)
        elif approach_id in [3, 4, 5]:
            all_flies, index_map, condition_names = load_scores_data(flies_data_paths, single=False,
                                                                     index_map=index_map,
                                                                     condition_names=condition_names)
        else:
            print("Unknown approach.")
            sys.exit(1)

    # 5) If needed, unify lengths for certain approaches
    if approach_id in [1, 3, 4, 5]:
        # all_flies might be a list of (T,B) arrays
        shapes = [f.shape for f in all_flies]
        if len(set(shapes)) != 1:
            data_3d = unify_lengths(all_flies, fill_value=0.0)
            all_flies = [data_3d[i] for i in range(data_3d.shape[0])]
    elif approach_id == 2:
        # If intervals, no unify needed
        pass

    # 6) Compute distance matrix
    dist_mat = compute_distance_matrix(all_flies, approach_id)

    # 7) Find the n most representative flies
    chosen = pick_n_representatives_simple(dist_mat, index_map, n)

    # 8) Print results
    print("\n=== Most Representative Flies ===")
    for rank, (global_idx, rep_metric) in enumerate(chosen, start=1):
        cond_idx, local_idx = index_map[global_idx]
        cond_name = condition_names[cond_idx] if cond_idx < len(condition_names) else "Unknown"

        # Convert representativeness to 0..100 scale
        # Example: 100 * (1 / (1 + dist_sum)) or any other formula
        rep_score = 100.0 * (1.0 / (1.0 + rep_metric))

        print(f"{rank}. Global Fly={global_idx}, Condition={cond_name}, "
              f"Local Fly={local_idx}, DistSum={rep_metric:.2f}, Score={rep_score:.2f}")


if __name__ == '__main__':
    main()
