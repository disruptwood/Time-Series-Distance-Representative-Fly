"""
distance_calculation/evaluate_all_pairs_no_sampling.py

All-Pairs, No Normalization, No Sampling Approach:
--------------------------------------------------
1) We parse the group name of each folder (e.g., "Females_Grouped").
2) We build an index -> group lookup: each fly i belongs to exactly one group,
   determined by which folder i came from (fly block).
3) We iterate over all fly pairs (i < j). If group[i] == group[j], it is an
   intra pair for that group. If group[i] != group[j], it is an inter pair
   for both group[i] and group[j].
4) For each group g, we compute the mean of all its intra distances and the
   mean of all its inter distances, then ratio = mean_intra / mean_inter.
5) If there's only one folder or only one group name, we compute an average
   distance among its flies if at least 2 exist; otherwise no ratio.

Debugging Aids:
---------------
- We print each group's total flies, a sample of indices, and how many inter-distances
  were collected for each group.

Usage Example:
--------------
from distance_calculation.evaluate_all_pairs_no_sampling import evaluate_distance_no_sampling

folder_paths = [
    r"D:\Data\Females_Grouped\Assa_Females_Grouped_Unknown_RigA_20220715T110325",
    r"D:\Data\Females_Grouped\Assa_Females_Grouped_Unknown_RigA_20220715T112156",
    r"D:\Data\Males_Single\Assa_Males_Single_Unknown_RigA_20220714T105158"
]
dist_mat = ... # NxN
evaluate_distance_no_sampling(folder_paths, dist_mat)
"""

import os
import numpy as np
from collections import defaultdict

def evaluate_distance_no_sampling(folder_paths, dist_mat):
    """
    Compare all pairs (i<j) in dist_mat. For each group:
      - Intra-distances: pairs (i,j) where group[i] == group[j]
      - Inter-distances: pairs (i,j) where group[i] == group[j] and group[j] != that group
                         or vice versa.

    If there's only one folder or only one group name among all folders:
      => if at least 2 flies, compute average distance, else no output.
    Otherwise => compute mean_intra, mean_inter, ratio for each group.

    Parameters
    ----------
    folder_paths : list of str
        Each folder path corresponds to a block of flies in dist_mat, in order.
        E.g. folder_paths[i] => block i of flies in dist_mat.
    dist_mat : np.ndarray, shape (N, N)
        The global distance matrix for all flies from these folders.
        The first block belongs to folder_paths[0], etc.

    Returns
    -------
    None (prints debug statements and results to stdout).
    """
    num_folders = len(folder_paths)
    N = dist_mat.shape[0]

    if num_folders == 0 or N < 2:
        print("No data or insufficient flies. Exiting.")
        return

    # 1) Figure out how many flies per folder (assuming uniform blocks)
    flies_per_folder = N // num_folders if num_folders > 0 else 0
    if flies_per_folder * num_folders != N:
        print(f"Warning: dist_mat size ({N}) not divisible by folder count ({num_folders}).")
        print("Indexing might be mismatched. Proceeding anyway.")
    #else:
        #print(f"Detected {num_folders} folder(s). {N} total flies => {flies_per_folder} per folder block.")

    # 2) Parse group name from each folder path
    def parse_group(folder_path):
        # Example: base="Assa_Males_Single_Unknown_RigA_20220714T105158" => "Males_Single"
        base = os.path.basename(folder_path)
        parts = base.split("_")
        if len(parts) >= 3:
            return parts[1] + "_" + parts[2]
        return "Unknown_Group"

    folder_groups = [parse_group(fp) for fp in folder_paths]

    # Build index -> group from block ranges
    index_to_group = {}
    start_idx = 0
    for i in range(num_folders):
        grp = folder_groups[i]
        end_idx = min(start_idx + flies_per_folder, N)
        for idx in range(start_idx, end_idx):
            index_to_group[idx] = grp
        start_idx += flies_per_folder

    # If there's effectively only 1 folder or 1 group name
    unique_groups = set(folder_groups)
    if num_folders == 1 or len(unique_groups) == 1:
        print("Single folder or single group => computing average distance among all flies in dist_mat.")
        if N < 2:
            print("Not enough flies for a pair. Exiting.")
            return
        triu = np.triu_indices(N, k=1)
        vals = dist_mat[triu]

        if len(vals):
            min_val, max_val = np.min(vals), np.max(vals)
            if max_val > min_val:  # Avoid division by zero
                norm_vals = 1 + 99 * (vals - min_val) / (max_val - min_val)  # Normalize to [1, 100]
                avg_dist = np.mean(norm_vals)
            else:
                avg_dist = 1.0  # If all distances are identical, set to lower bound
        else:
            avg_dist = 1.0  # If no distances exist, default to 1.0

        print(f"Normalized Average Distance: {avg_dist:.4f}")

    # 3) We create group_intra[g] and group_inter[g]
    group_intra = defaultdict(list)
    group_inter = defaultdict(list)

    # 4) For debug: track which indices belong to which group
    group_indices_map = defaultdict(list)
    for idx in range(N):
        g = index_to_group[idx]
        group_indices_map[g].append(idx)

    # 5) Build all pairs (i<j) once, add to intra if same group, else to inter for each group
    for i in range(N - 1):
        gi = index_to_group.get(i, None)
        for j in range(i+1, N):
            gj = index_to_group.get(j, None)
            if gi is None or gj is None:
                continue
            val = dist_mat[i, j]
            if gi == gj:
                group_intra[gi].append(val)
            else:
                group_inter[gi].append(val)
                group_inter[gj].append(val)

    # Debug: print group compositions
    #print("\n=== Debug: Group Composition ===")
    #for g, inds in group_indices_map.items():
        #print(f"Group '{g}' => #flies={len(inds)}, example={inds[:5]}")

    # Debug: how many inter-distances each group has
    #print("\n=== Debug: Inter Distances Collected ===")
    for g in group_indices_map.keys():
        vals = group_inter[g]
        #print(f"Group '{g}' => # inter-distances={len(vals)}  Sample={vals[:10]}")

    # 6) For each group, compute mean_intra, mean_inter => ratio
    all_ratios = []
    group_names_sorted = sorted(list(group_indices_map.keys()))
    #print("\n=== All-Pairs, No Sampling: Intra vs. Inter Distances by Group ===")
    for g in group_names_sorted:
        ilist = group_intra[g]
        mean_intra = np.mean(ilist) if len(ilist) > 0 else np.nan

        elist = group_inter[g]
        mean_inter = np.mean(elist) if len(elist) > 0 else np.nan

        if np.isnan(mean_intra) or np.isnan(mean_inter) or mean_inter == 0:
            ratio = np.inf
        else:
            ratio = mean_intra / mean_inter

        #print(f"Group={g}, mean_intra={mean_intra:.4f}, mean_inter={mean_inter:.4f}, ratio={ratio:.4f}")
        if ratio not in [np.inf, np.nan] and not np.isnan(ratio):
            all_ratios.append(ratio)

    if len(all_ratios) > 0:
        avg_ratio = np.mean(all_ratios)
        print(f"\nOverall average ratio (intra/inter): {avg_ratio:.4f}")
