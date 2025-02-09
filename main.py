# Main for running GUI for fly representativity calculation
import sys

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
from distance_calculation.approach3_markov_pretrain import pretrain_and_build_dist_matrix, normalize_all_flies
from distance_calculation.pick_n_representative import pick_n_representatives_simple
from distance_calculation.evaluate_method import evaluate_distance_no_sampling

def main():
    try:
        # 1) Choose single or multiple
        mode = pick_input_mode()
        print(f"Mode selected: {mode}")

        # 2) Choose approach
        approach_id = pick_approach()
        print(f"Approach ID selected: {approach_id}")

        # 3) Choose number of representatives
        n = pick_n_representatives()
        print(f"Number of representatives: {n}")

        # 4) Collect data
        if mode == 'single':
            mat_files = pick_mat_files_single_condition()
            print(f"Selected .mat files: {mat_files}")
            flies_data_paths = gather_mat_files_single_condition(mat_files)
            print(f"Gathered flies data paths: {flies_data_paths}")

            if approach_id == 1:
                all_flies, index_map, condition_names = load_postprocessed_data(flies_data_paths, single=True)
            elif approach_id == 2:
                all_flies, index_map, condition_names = load_intervals_data(flies_data_paths, single=True)
            elif approach_id in [3, 4, 5]:
                all_flies, index_map, condition_names = load_scores_data(flies_data_paths, single=True)
                all_flies = normalize_all_flies(all_flies)
            else:
                print("Unknown approach.")
                sys.exit(1)

            all_condition_files = [flies_data_paths]

        else:
            folder_list = pick_multiple_folders_for_conditions()
            print(f"Selected folders: {folder_list}")

            # gather_mat_files_multiple_condition now returns (list_of_lists_of_files, condition_names)
            all_condition_files, condition_names = gather_mat_files_multiple_condition(folder_list)

            if approach_id == 1:
                all_flies, index_map, condition_names = load_postprocessed_data(all_condition_files, single=False, condition_names=condition_names)
            elif approach_id == 2:
                all_flies, index_map, condition_names = load_intervals_data(all_condition_files, single=False, condition_names=condition_names)
            elif approach_id in [3, 4, 5]:
                all_flies, index_map, condition_names = load_scores_data(all_condition_files, single=False, condition_names=condition_names)
            else:
                print("Unknown approach.")
                sys.exit(1)

        print("Data loading completed.")
        print(f"Index map: {index_map}")
        print(f"Condition names: {condition_names}")

        # 5) unify lengths for certain approaches
        if approach_id in [1, 3, 4, 5]:
            shapes = [f.shape for f in all_flies]
            if len(set(shapes)) != 1:
                data_3d = unify_lengths(all_flies, fill_value=0.0)
                all_flies = [data_3d[i] for i in range(data_3d.shape[0])]
        print("Data unified for consistent lengths.")

        '''' Debugging output for first fly
        first_global_idx = 0
        first_local_idx = index_map[first_global_idx][1]
        print(f"First Fly Global Index: {first_global_idx}")
        print(f"First Fly Local Index: {first_local_idx}")

        if approach_id == 1:
            print(f"First 10 values from postprocessed binary: {all_flies[0][:10]}")
        elif approach_id == 2:
            print(f"First 10 intervals: {all_flies[0][:10]}")
        elif approach_id in [3, 4, 5]:
            print(f"First 10 values from scores: {all_flies[0][:10]}")
        '''''
        if approach_id == 3:
            # If we loaded data as a flat list 'all_flies', reorganize by condition
            # i.e. all_flies_scores[cond_idx] = list of flies for that condition
            num_conditions = len(condition_names)
            all_flies_scores = [[] for _ in range(num_conditions)]
            for global_idx, (cond_idx, local_idx) in index_map.items():
                all_flies_scores[cond_idx].append(all_flies[global_idx])
            for cond_idx in range(len(all_flies_scores)):
                all_flies_scores[cond_idx] = normalize_all_flies(all_flies_scores[cond_idx])

            # We call pretrain function that:
            #   1) trains & saves HMM for each condition if not present
            #   2) builds a global distance matrix
            dist_mat = pretrain_and_build_dist_matrix(
                all_flies_scores=all_flies_scores,
                condition_names=condition_names,
                index_map=index_map,
                all_condition_files=all_condition_files,
                n_hidden_states=8,
                n_iter=150,
                tol=1e-3,
                covariance_type='full'
            )
            print("HMM-based distance matrix built.")

        else:
            # 6) Compute distance matrix
            dist_mat = compute_distance_matrix(all_flies, approach_id)
            print("Distance matrix computed.")

        # 7) Find the n most representative flies
        chosen = pick_n_representatives_simple(dist_mat, index_map, n)

        # 8) Print results and normalize DistSum for representativity scores
        dist_sums = [rep_metric for _, rep_metric in chosen]
        min_sum = min(dist_sums) if dist_sums else 0.0
        max_sum = max(dist_sums) if dist_sums else 1.0
        range_sum = max_sum - min_sum
        if abs(range_sum) < 1e-12:
            range_sum = 1.0  # fallback to avoid divide-by-zero

        print("\n=== Most Representative Flies ===")
        for rank, (global_idx, rep_metric) in enumerate(chosen, start=1):
            cond_idx, local_idx = index_map[global_idx]
            cond_name = condition_names[cond_idx] if cond_idx < len(condition_names) else "Unknown"
            # Shift & scale dist_sum to [0..1]
            shifted_dist = rep_metric - min_sum
            scaled_dist = shifted_dist / range_sum  # 0.0 => best, 1.0 => worst

            # Normalize representativity score
            rep_score = 100.0 * (1.0 - scaled_dist)  # Smaller DistSum -> Higher Score

            print(f"{rank}. Global Fly={global_idx}, Condition={cond_name}, "
                  f"Local Fly={local_idx}, DistSum={rep_metric:.2f}, Score={rep_score:.2f}")
        #print(dist_mat)
        # Intra Iner Ratio
        evaluate_distance_no_sampling(condition_names, dist_mat)


    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
