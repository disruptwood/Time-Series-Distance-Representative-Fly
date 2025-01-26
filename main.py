import os
import glob
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import pairwise_distances


def pick_input_mode():
    """
    Opens a small GUI window with two buttons:
      'Single Condition' => pick multiple .mat files for one condition
      'Multiple Conditions' => pick one top-level folder containing subfolders (each subfolder is a condition).
    Returns either 'single' or 'multiple'.
    """
    root = tk.Tk()
    root.title("Select Input Mode")
    selection = []

    def choose_single():
        selection.append("single")
        root.destroy()

    def choose_multiple():
        selection.append("multiple")
        root.destroy()

    label = tk.Label(root, text="Do you want to load a single condition or multiple conditions?")
    label.pack(padx=20, pady=10)

    btn_single = tk.Button(root, text="Single Condition", command=choose_single)
    btn_single.pack(side=tk.LEFT, padx=20, pady=10)

    btn_multiple = tk.Button(root, text="Multiple Conditions", command=choose_multiple)
    btn_multiple.pack(side=tk.RIGHT, padx=20, pady=10)

    root.mainloop()

    if not selection:
        raise RuntimeError("No selection made.")
    return selection[0]


def pick_mat_files():
    """
    Opens a dialog to pick multiple .mat files at once for a single condition.
    Returns a list of absolute paths.
    """
    root = tk.Tk()
    root.withdraw()
    chosen = filedialog.askopenfilenames(
        title='Select .mat Files (One Condition)',
        filetypes=[('MAT files', '*.mat')]
    )
    if not chosen:
        raise RuntimeError("No .mat files selected for single condition.")
    return [os.path.abspath(p) for p in chosen]


def pick_folder_for_conditions():
    """
    Opens a dialog to pick ONE top-level folder that contains subfolders.
    Each subfolder is treated as a condition.
    Returns the absolute path to that top-level folder.
    """
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Top-Level Folder (Multiple Conditions)")
    if not folder:
        raise RuntimeError("No folder selected for multiple conditions.")
    return os.path.abspath(folder)


def find_condition_folders(top_folder):
    """
    Lists immediate subfolders of 'top_folder' (ignoring '.'/'..').
    Each subfolder is considered a separate condition.
    Returns a list of absolute paths to those condition subfolders.
    """
    subfolders = []
    with os.scandir(top_folder) as entries:
        for e in entries:
            if e.is_dir() and not e.name.startswith('.'):
                subfolders.append(os.path.abspath(e.path))
    if not subfolders:
        raise RuntimeError(f"No subfolders found in {top_folder}.")
    return sorted(subfolders)


def gather_behavior_files_in_folder(condition_folder):
    """
    For a condition folder, gather all 'scores_*.mat' (no further subfolders).
    Returns a list of .mat file paths. (Assumes each file is one behavior.)
    """
    pattern = os.path.join(condition_folder, "scores_*.mat")
    found = glob.glob(pattern)
    if not found:
        # Optionally look into subfolders if user has nested structure
        # If not needed, remove this part.
        sub_mat = []
        with os.scandir(condition_folder) as entries:
            for e in entries:
                if e.is_dir():
                    sub_files = glob.glob(os.path.join(e.path, "scores_*.mat"))
                    sub_mat.extend(sub_files)
        found = sub_mat

    if not found:
        raise RuntimeError(f"No 'scores_*.mat' files found in {condition_folder}.")
    return sorted(os.path.abspath(f) for f in found)


def load_condition(mat_file_list):
    """
    Loads multiple .mat files, each containing one behavior's postprocessed data
    for the same set of flies. Returns a list-of-flies, where each fly is a (T, B) array.
      B = len(mat_file_list)
      T can vary per fly, so we store each fly as an array with its own length.
    Also returns the number of flies (for sanity checks).
    """
    if not mat_file_list:
        raise RuntimeError("No .mat files provided for a condition.")

    # Load first file to see # of flies
    data0 = loadmat(mat_file_list[0])['allScores']
    if 'postprocessed' not in data0.dtype.names:
        raise RuntimeError(f"No 'postprocessed' field in first file {mat_file_list[0]}")
    pp0 = data0['postprocessed'][0, 0]
    num_flies = max(pp0.shape)

    # We'll build for each fly => list of columns
    fly_behaviors = [[] for _ in range(num_flies)]
    max_lengths = [0]*num_flies  # track max T per fly

    for mf in mat_file_list:
        mat_data = loadmat(mf)['allScores']
        if 'postprocessed' not in mat_data.dtype.names:
            raise RuntimeError(f"No 'postprocessed' field in file {mf}")
        postprocessed = mat_data['postprocessed'][0, 0]

        t_starts = mat_data['tStart'][0, 0][0]
        t_ends   = mat_data['tEnd'][0, 0][0]

        # figure out how to index each fly from postprocessed
        rows, cols = postprocessed.shape
        if rows == num_flies:
            get_fly_pp = lambda i: postprocessed[i, 0]
        elif cols == num_flies:
            get_fly_pp = lambda i: postprocessed[0, i]
        else:
            raise RuntimeError(f"Mismatch #flies vs. postprocessed shape {postprocessed.shape} in file {mf}.")

        for fly_i in range(num_flies):
            arr = get_fly_pp(fly_i).ravel()
            start_idx = t_starts[fly_i] - 1
            end_idx   = t_ends[fly_i]
            if end_idx > len(arr):
                raise RuntimeError(f"Fly {fly_i} in file {mf} has invalid tEnd={end_idx}, arr length={len(arr)}")

            valid_slice = arr[start_idx:end_idx]
            fly_behaviors[fly_i].append(valid_slice)
            if len(valid_slice) > max_lengths[fly_i]:
                max_lengths[fly_i] = len(valid_slice)

    # Now unify each fly into (T, B)
    # B = len(mat_file_list)
    # T can vary by fly. We'll store them as separate arrays.
    all_flies_this_condition = []
    for i in range(num_flies):
        T = max_lengths[i]
        B = len(mat_file_list)
        mat = np.full((T, B), np.nan, dtype=float)
        for b_idx, col in enumerate(fly_behaviors[i]):
            mat[:len(col), b_idx] = col
        all_flies_this_condition.append(mat)

    return all_flies_this_condition, num_flies


def load_multiple_conditions(top_folder):
    """
    Treats each immediate subfolder of 'top_folder' as a separate condition.
    For each subfolder, gathers 'scores_*.mat', loads them, and accumulates the
    resulting flies in all_flies. Also builds an index_map: (condition_idx, fly_idx_in_that_condition).
    Returns:
      all_flies: list of (T, B) arrays for all conditions
      index_map: parallel list of (cond_idx, fly_idx)
      condition_names: list of condition folder names
    """
    condition_folders = find_condition_folders(top_folder)
    all_flies = []
    index_map = []
    condition_names = []

    for cond_idx, cond_folder in enumerate(condition_folders):
        mat_file_list = gather_behavior_files_in_folder(cond_folder)
        flies_this_cond, num_flies = load_condition(mat_file_list)
        # Append them
        for fly_i, arr in enumerate(flies_this_cond):
            all_flies.append(arr)
            index_map.append((cond_idx, fly_i))
        condition_names.append(os.path.basename(cond_folder))

    return all_flies, index_map, condition_names


def unify_lengths(all_flies, fill_value=0.0):
    """
    Optional helper to handle variable T across flies by padding to the max length.
    Replaces NaN with fill_value. This ensures each fly array has the same shape.
    Returns a single 3D array of shape (N, T_max, B).
    """
    N = len(all_flies)
    max_T = max(arr.shape[0] for arr in all_flies)
    B = all_flies[0].shape[1] if all_flies else 0

    output = np.full((N, max_T, B), fill_value, dtype=float)
    for i, arr in enumerate(all_flies):
        t_len, b_len = arr.shape
        output[i, :t_len, :b_len] = np.nan_to_num(arr, nan=fill_value)
    return output


def build_distance_matrix(all_flies, metric='euclidean'):
    """
    Example pairwise distance after flattening each fly's (T x B).
    If arrays differ in shape, you need to unify them first.
    """
    # Flatten
    # Verify all have the same shape or unify them
    shapes = [x.shape for x in all_flies]
    if len(set(shapes)) != 1:
        # Not all the same shape => unify
        data_3d = unify_lengths(all_flies, fill_value=0.0)
        X = data_3d.reshape((data_3d.shape[0], -1))
    else:
        X = np.array([f.ravel() for f in all_flies])

    return pairwise_distances(X, metric=metric)


def main():
    mode = pick_input_mode()
    if mode == 'single':
        # Single condition => pick .mat files directly
        mat_file_list = pick_mat_files()
        all_flies, num_flies = load_condition(mat_file_list)
        # Index map => single condition
        index_map = [(0, i) for i in range(num_flies)]
        # Example distance
        dist_mat = build_distance_matrix(all_flies, metric='euclidean')
        print(f"Single condition => {num_flies} flies, {len(mat_file_list)} behaviors.")
        print("Distance matrix shape:", dist_mat.shape)

    else:
        # Multiple conditions => pick top-level folder
        top_folder = pick_folder_for_conditions()
        all_flies, index_map, condition_names = load_multiple_conditions(top_folder)
        print(f"Found {len(condition_names)} conditions. Total flies={len(all_flies)}.")
        # Example distance
        dist_mat = build_distance_matrix(all_flies, metric='euclidean')
        print("Distance matrix shape:", dist_mat.shape)


if __name__ == '__main__':
    main()
