import numpy as np

def unify_lengths(all_flies, fill_value=0.0):
    """
    Pads (T, B) arrays to the same length across all flies.
    Handles missing frames by filling them with a specified value.

    Args:
        all_flies (list of np.ndarray): List of (T, B) arrays where T is the time dimension and B is the behavior dimension.
        fill_value (float): The value to fill for missing frames. Default is 0.0.

    Returns:
        np.ndarray: A 3D array of shape (N, max_T, B), where:
            - N is the number of flies.
            - max_T is the maximum time length across all flies.
            - B is the number of behaviors per fly.
    """
    if not all_flies:
        raise RuntimeError("Empty input list for all_flies.")

    # Determine dimensions
    N = len(all_flies)
    max_T = max(f.shape[0] for f in all_flies)  # Maximum time length
    B = all_flies[0].shape[1] if all_flies else 0  # Behavior dimension

    # Create an output array with the desired shape, filled with the fill_value
    out = np.full((N, max_T, B), fill_value, dtype=float)

    # Populate the array with data from each fly, padding as necessary
    for i, f in enumerate(all_flies):
        t_len, _ = f.shape
        out[i, :t_len, :] = np.nan_to_num(f, nan=fill_value)

    return out
