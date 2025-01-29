# data_loading/gather_mat_files_single.py

import os

def gather_mat_files_single_condition(mat_files):
    """
    Takes a list of .mat file paths (selected from a single condition) and ensures they are valid.
    Returns the sorted, absolute paths to the files.

    Args:
        mat_files (list of str): List of paths to .mat files.

    Returns:
        list of str: Sorted absolute paths to the valid .mat files.
    """
    # Validate input files
    if not mat_files:
        raise RuntimeError("No .mat files provided for single condition.")

    # Check that all provided paths are .mat files and exist
    valid_files = []
    for file_path in mat_files:
        if not os.path.isfile(file_path):
            raise RuntimeError(f"File does not exist: {file_path}")
        if not file_path.lower().endswith(".mat"):
            raise RuntimeError(f"Invalid file format (not .mat): {file_path}")
        valid_files.append(os.path.abspath(file_path))

    # Sort files for consistency
    valid_files.sort()
    return valid_files
