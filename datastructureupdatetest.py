# Possible dta structure update
#1: Import Libraries
import os
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from data_loading.gather_mat_files_multiple import gather_mat_files_multiple_condition
from data_loading.load_postprocessed import load_postprocessed_data


#2: Data Loading Function
def load_behavior_data(base_folder):
    """
    Load postprocessed behavior data and extract behavior names from filenames.

    Args:
        base_folder (str): The base folder containing .mat files.

    Returns:
        tuple:
            - all_flies_post (list): Postprocessed data per fly.
            - post_index_map (dict): Index mapping.
            - condition_names (list): Condition names.
            - behavior_names_per_condition (list): Extracted behavior names from filenames.
    """
    # Gather .mat files for all conditions
    all_condition_files, condition_names = gather_mat_files_multiple_condition([base_folder])

    # Load postprocessed data
    all_flies_post, post_index_map, _ = load_postprocessed_data(
        all_condition_files,
        single=False,
        condition_names=condition_names
    )

    # Extract behavior names from filenames
    behavior_names_per_condition = []
    for cond_files in all_condition_files:
        names_for_this_cond = []
        for filepath in cond_files:
            base = os.path.basename(filepath)  # e.g. "scores_Grooming.mat"
            no_ext = os.path.splitext(base)[0]  # e.g. "scores_Grooming"
            parts = no_ext.split("_", 1)  # split once on first '_'
            behavior_name = parts[1] if len(parts) == 2 else no_ext  # Extract behavior name
            names_for_this_cond.append(behavior_name)
        behavior_names_per_condition.append(names_for_this_cond)

    return all_flies_post, post_index_map, condition_names, behavior_names_per_condition


#3: Compute Behavior Sums
base_folder = r"D:\\behavior_ethogram_project_Ilya"
all_flies_post, post_index_map, condition_names, behavior_names_per_condition = load_behavior_data(base_folder)

# Prepare table
table = PrettyTable()
table.field_names = ["Condition", "Behavior Name", "Fly ID", "Sum of Postprocessed Values"]

global_fly_id = 0
for global_fly_idx, (fly_data, (condition_idx, local_fly_idx)) in enumerate(post_index_map.items()):
    condition_name = condition_names[condition_idx]
    behavior_names = behavior_names_per_condition[condition_idx]
    fly_data = np.asarray(all_flies_post[global_fly_idx])  # Ensure it's an array
    if fly_data.ndim == 2 and fly_data.shape[1] == len(behavior_names):
        for behavior_idx, behavior_name in enumerate(behavior_names):
            behavior_sum = np.nansum(fly_data[:, behavior_idx])
            table.add_row([condition_name, behavior_name, global_fly_idx, behavior_sum])
    else:
        behavior_sum = np.nansum(fly_data)  # Handle 1D case
        table.add_row([condition_name, "Unknown", global_fly_idx, behavior_sum])

print(table)
