import os
import pickle

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # For a progress bar

####
# 1) Function to Train a GaussianHMM for One Fly and Save the Model
####

def train_and_save_gaussian_hmm(fly_data, fly_idx, save_folder,
                                n_hidden_states=5, n_iter=100, tol=1e-3,
                                covariance_type='full'):
    """
    Trains a GaussianHMM for a single fly's scores data, then saves
    the resulting model parameters via pickle in the specified folder.

    Parameters:
      fly_data       : a 2D NumPy array shape (T, B).
      fly_idx        : the index of this fly (int).
      save_folder    : path to the folder where the model's parameters will be saved.
      n_hidden_states: number of hidden states for the HMM.
      n_iter         : max number of EM iterations for training.
      tol            : tolerance for convergence.
      covariance_type: 'full', 'diag', 'tied', or 'spherical'.

    Output:
      A pickle file named 'fly_{fly_idx}.pkl' in save_folder containing a dict with
      the model parameters: startprob, transmat, means, covars, etc.
    """
    # Create folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Train HMM
    model = hmm.GaussianHMM(n_components=n_hidden_states,
                            n_iter=n_iter,
                            tol=tol,
                            covariance_type=covariance_type,
                            random_state=42,
                            verbose=False)
    model.fit(fly_data)  # (T, B)

    # Extract parameters
    hmm_params = {
        "n_components": model.n_components,
        "covariance_type": model.covariance_type,
        "startprob": model.startprob_,
        "transmat": model.transmat_,
        "means": model.means_,
        "covars": model.covars_,
        # Optionally store additional info
        # "monitor": model.monitor_  # if you want the training log
    }

    # Save to pickle
    output_path = os.path.join(save_folder, f"fly_{fly_idx}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(hmm_params, f)


####
# 2) Code to Train & Save Models for All Flies (Scores Data)
####

def train_and_save_hmms_for_all_flies(all_flies_scores, save_folder,
                                      n_hidden_states=5, n_iter=100, tol=1e-3,
                                      covariance_type='full'):
    """
    Iterates over the list of flies (all_flies_scores), checks if each fly's model
    is already saved. If not, trains a new model and saves it.

    Parameters:
      all_flies_scores : list of 2D arrays (T, B) for each fly.
      save_folder      : folder to save each fly's parameters.
      n_hidden_states  : number of hidden states in the HMM.
      n_iter           : max number of EM iterations.
      tol              : EM convergence threshold.
      covariance_type  : type of covariance matrix to use.

    This function prints progress so that if it is stopped, the next run
    will skip already-saved models.
    """
    os.makedirs(save_folder, exist_ok=True)

    N = len(all_flies_scores)
    for fly_idx in range(N):
        output_path = os.path.join(save_folder, f"fly_{fly_idx}.pkl")
        if os.path.exists(output_path):
            print(f"Fly {fly_idx}: model already saved. Skipping.")
            continue

        # Train and save
        fly_data = all_flies_scores[fly_idx]
        print(f"Training model for Fly {fly_idx}...")
        train_and_save_gaussian_hmm(fly_data, fly_idx, save_folder,
                                    n_hidden_states=n_hidden_states,
                                    n_iter=n_iter, tol=tol,
                                    covariance_type=covariance_type)
        print(f"...Saved model for Fly {fly_idx} (file: {output_path}).")



####
# 3) Build Distance Matrix from Saved Models (No Re-training)
####

def load_hmm_model_params(filepath):
    """
    Loads HMM parameters from a pickle file and reconstructs a dictionary
    of the needed parameters. This does NOT return a hmm.GaussianHMM instance
    by default, but you can reconstruct one if needed.
    """
    with open(filepath, "rb") as f:
        model_params = pickle.load(f)
    return model_params

def sym_kl_transitions(paramsA, paramsB):
    """
    Symmetrized KL divergence on transition matrices of two HMM parameter dicts.
    """
    import numpy as np
    from scipy.special import rel_entr

    transA = paramsA["transmat"] + 1e-9
    transB = paramsB["transmat"] + 1e-9
    kl_ab = np.sum(rel_entr(transA, transB))
    kl_ba = np.sum(rel_entr(transB, transA))
    return 0.5 * (kl_ab + kl_ba)

def gaussian_kl(mu1, Sigma1, mu2, Sigma2):
    """
    KL divergence for full-covariance Gaussians.
    """
    import numpy as np
    d = mu1.shape[0]
    invSigma2 = np.linalg.inv(Sigma2)
    det1 = np.linalg.det(Sigma1)
    det2 = np.linalg.det(Sigma2)
    ratio = np.log((det2 + 1e-12)/(det1 + 1e-12))
    trace_term = np.trace(invSigma2 @ Sigma1)
    diff = (mu2 - mu1).reshape(-1, 1)
    mahal = float(diff.T @ invSigma2 @ diff)
    return 0.5 * (ratio - d + trace_term + mahal)

def sym_kl_gaussian_emissions(paramsA, paramsB):
    """
    Compare emissions of two HMM parameter dicts. Summation of sym KL
    for each state pair. This assumes same number of components.
    """
    import numpy as np

    meansA  = paramsA["means"]
    covarsA = paramsA["covars"]
    meansB  = paramsB["means"]
    covarsB = paramsB["covars"]

    n_states = meansA.shape[0]
    total_kl = 0.0
    for i in range(n_states):
        kl_ab = gaussian_kl(meansA[i], covarsA[i], meansB[i], covarsB[i])
        kl_ba = gaussian_kl(meansB[i], covarsB[i], meansA[i], covarsA[i])
        total_kl += 0.5 * (kl_ab + kl_ba)
    return total_kl

def hmm_distance(paramsA, paramsB):
    """
    Distance measure between two sets of HMM parameters.
    Combines sym. KL for transitions + sym. KL for emissions.
    """
    return sym_kl_transitions(paramsA, paramsB) + sym_kl_gaussian_emissions(paramsA, paramsB)

def build_distance_matrix_from_models(save_folder, n_flies):
    """
    Builds an n_flies × n_flies distance matrix using previously saved
    HMM model parameters. The distance measure is user-defined (here, a
    sum of symmetrized KL divergences on transitions + emissions).

    Parameters:
      save_folder : the folder containing 'fly_{idx}.pkl' files.
      n_flies     : how many flies total are expected.

    Returns:
      dist_mat: a (n_flies, n_flies) numpy array of distances.
    """
    import numpy as np
    dist_mat = np.zeros((n_flies, n_flies), dtype=float)

    # Pre-load all HMM parameter dicts
    all_params = [None]*n_flies
    for i in range(n_flies):
        filepath = os.path.join(save_folder, f"fly_{i}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing HMM model file for fly {i} (expected {filepath}).")
        all_params[i] = load_hmm_model_params(filepath)

    # Compute upper-triangle distances
    for i in range(n_flies):
        for j in range(i+1, n_flies):
            dist_val = hmm_distance(all_params[i], all_params[j])
            dist_mat[i, j] = dist_val
            dist_mat[j, i] = dist_val

    return dist_mat

def train_gaussian_hmm(fly_data, n_hidden_states=5, n_iter=100, tol=1e-3, covariance_type='full'):
    """
    Train a GaussianHMM on fly_data, return its parameters as a dict (no pickling here).
    """
    model = hmm.GaussianHMM(n_components=n_hidden_states,
                            n_iter=n_iter,
                            tol=tol,
                            covariance_type=covariance_type,
                            random_state=42,
                            verbose=False)
    model.fit(fly_data)  # fly_data is (T,B)
    return {
        "n_components": model.n_components,
        "covariance_type": model.covariance_type,
        "startprob": model.startprob_,
        "transmat": model.transmat_,
        "means": model.means_,
        "covars": model.covars_,
    }

def train_and_save_hmms_for_one_condition(condition_scores, cond_folder,
                                          n_hidden_states=5, n_iter=100, tol=1e-3,
                                          covariance_type='full'):
    """
    Train an HMM for each fly in a single condition and save to cond_folder/HMMModels.
    Skip if existing model file is found for that fly.
    Returns a list of the loaded (or newly trained) HMM parameter dicts for all flies.
    """
    save_folder = os.path.join(cond_folder, "HMMModels")
    os.makedirs(save_folder, exist_ok=True)

    num_flies = len(condition_scores)
    all_params = []
    for fly_idx in range(num_flies):
        out_path = os.path.join(save_folder, f"fly_{fly_idx}.pkl")
        if os.path.exists(out_path):
            # Already trained => load existing
            print(f"Fly {fly_idx} in {cond_folder} already has model. Loading existing.")
            params = load_hmm_model_params(out_path)
            all_params.append(params)
            continue

        # Otherwise, train a new HMM
        print(f"Training HMM for Fly {fly_idx} in {cond_folder}...")
        fly_data = condition_scores[fly_idx]
        hmm_params = train_gaussian_hmm(fly_data, n_hidden_states=n_hidden_states,
                                        n_iter=n_iter, tol=tol, covariance_type=covariance_type)
        # Save to pickle
        with open(out_path, "wb") as f:
            pickle.dump(hmm_params, f)
        print(f"...Saved model for Fly {fly_idx} to {out_path}.")
        all_params.append(hmm_params)

    return all_params


def build_distance_matrix_for_condition(cond_folder, condition_scores):
    """
    Loads each fly's HMM from cond_folder/HMMModels/fly_{idx}.pkl, builds a distance matrix
    among those flies. Returns a (num_flies x num_flies) numpy array.
    """
    save_folder = os.path.join(cond_folder, "HMMModels")
    num_flies = len(condition_scores)
    dist_mat = np.zeros((num_flies, num_flies), dtype=float)

    # Load all HMM parameters
    all_params = []
    for fly_idx in range(num_flies):
        in_path = os.path.join(save_folder, f"fly_{fly_idx}.pkl")
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"No HMM model found for fly {fly_idx} at {in_path}")
        params = load_hmm_model_params(in_path)
        all_params.append(params)

    # Fill pairwise distances (upper triangle)
    for i in range(num_flies):
        for j in range(i + 1, num_flies):
            dval = hmm_distance(all_params[i], all_params[j])
            dist_mat[i, j] = dval
            dist_mat[j, i] = dval

    return dist_mat


def pretrain_and_build_dist_matrix(all_flies_scores, condition_names, index_map,
                                   all_condition_files,
                                   n_hidden_states=5, n_iter=100, tol=1e-3,
                                   covariance_type='full'):
    """
    Main entry for approach=3. For each condition (folder):
      1) Train or load existing HMM model for each fly => store param dict in memory.
      2) Build a full NxN distance matrix across *all* conditions and flies:
         we do cross-distances among flies in different conditions.

    all_flies_scores: list of lists. all_flies_scores[c] => list of (T,B) arrays for condition c
    condition_names: list of condition folder base names
    index_map: global_idx -> (cond_idx, local_fly_idx)
    all_condition_files: same shape as condition_names, each is a list of .mat files for that condition
    n_hidden_states, n_iter, tol, covariance_type: HMM parameters

    Returns:
      dist_mat_global: NxN distance matrix (N=total flies across all conditions).
                       Indices correspond to the same global ordering as:
                         0..(count of cond0 flies)-1, then cond1, etc.
    """
    num_conditions = len(condition_names)
    # We'll store the HMM param dicts for each fly in each condition
    # all_params_by_condition[c] = [param_for_fly0, param_for_fly1, ...]
    all_params_by_condition = []
    condition_offsets = []
    total_flies = 0

    # 1) Train or load existing HMMs for each condition
    for cond_idx in range(num_conditions):
        cond_name = condition_names[cond_idx]
        if not all_condition_files[cond_idx]:
            # no .mat files
            print(f"No files for condition {cond_name}. Skipping.")
            all_params_by_condition.append([])
            condition_offsets.append(total_flies)
            continue

        cond_folder = os.path.dirname(all_condition_files[cond_idx][0])  # e.g. "D:\...\MyCondition"
        condition_scores_list = all_flies_scores[cond_idx]  # list of flies for this condition
        if not condition_scores_list:
            print(f"No flies for condition {cond_name}. Skipping.")
            all_params_by_condition.append([])
            condition_offsets.append(total_flies)
            continue

        # Train or load
        params_for_this_cond = train_and_save_hmms_for_one_condition(
            condition_scores_list, cond_folder,
            n_hidden_states=n_hidden_states,
            n_iter=n_iter, tol=tol,
            covariance_type=covariance_type
        )
        all_params_by_condition.append(params_for_this_cond)
        condition_offsets.append(total_flies)
        total_flies += len(params_for_this_cond)

    # 2) Create the full NxN distance matrix
    print(f"Building a cross-condition distance matrix with total {total_flies} flies.")
    dist_mat_global = np.zeros((total_flies, total_flies), dtype=float)

    # We'll fill it by iterating over all conditions c1, c2, all flies i, j
    for c1 in range(num_conditions):
        params_c1 = all_params_by_condition[c1]
        offset_c1 = condition_offsets[c1]
        for i, param_i in enumerate(params_c1):
            global_i = offset_c1 + i

            for c2 in range(num_conditions):
                params_c2 = all_params_by_condition[c2]
                offset_c2 = condition_offsets[c2]
                for j, param_j in enumerate(params_c2):
                    global_j = offset_c2 + j

                    if global_j <= global_i:
                        continue  # skip lower triangle (avoid duplicates), fill once
                    # compute cross distance
                    dval = hmm_distance(param_i, param_j)
                    dist_mat_global[global_i, global_j] = dval
                    dist_mat_global[global_j, global_i] = dval

    print(f"Final cross-condition distance matrix shape: {dist_mat_global.shape}")
    return dist_mat_global

def normalize_fly_data(fly_data):
    """
    Normalize a single fly's data (T x B) so that each feature dimension
    has zero mean and unit variance, across the T frames.

    fly_data: shape (T, B)
    Returns: shape (T, B) with normalized columns.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(fly_data)  # shape (T,B), zero-mean, unit-variance

def normalize_all_flies(all_flies_scores):
    """
    Applies standard scaling to every fly in the list.
    all_flies_scores is a list of arrays, each shape (T, B).

    Returns a new list of the same shape, but normalized.
    """
    normalized = []
    for fly_data in all_flies_scores:
        norm_data = normalize_fly_data(fly_data)
        normalized.append(norm_data)
    return normalized