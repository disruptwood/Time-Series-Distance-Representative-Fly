# data_loading/gather_mat_files_multiple.py

import os
import glob
import logging

def gather_mat_files_multiple_condition(top_level_folders):
    """
    For each folder in 'top_level_folders', recursively finds subfolders that
    contain 'scores_*.mat' files. Each subfolder with those files is treated
    as a separate condition.

    Returns:
      all_condition_files (list of list of str):
        Each element is a list of .mat file paths for one condition.
      condition_names (list of str):
        Parallel list of condition names for each subfolder that was found.
    """
    if not top_level_folders:
        raise RuntimeError("No folders provided for multiple conditions.")

    all_condition_files = []
    condition_names = []

    for root_folder in top_level_folders:
        # Walk the directory tree
        for subdir, dirs, files in os.walk(root_folder):
            pattern = os.path.join(subdir, "scores_*.mat")
            found = glob.glob(pattern)
            if found:
                # Sort for consistency
                found = sorted(os.path.abspath(f) for f in found)
                # Use the subfolder name as the condition name
                cond_name = os.path.basename(os.path.normpath(subdir))

                all_condition_files.append(found)
                condition_names.append(cond_name)

    if not all_condition_files:
        raise RuntimeError("No 'scores_*.mat' files found in any subfolder of the selected folders.")

    return all_condition_files, condition_names