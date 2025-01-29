import numpy as np
from hmmlearn import hmm
from scipy.special import rel_entr

def approach3_markov_distance(seqA, seqB, n_hidden_states=5, n_iter=100):
    """
    Computes a distance between two continuous sequences using GaussianHMMs.
    The distance is a combination of:
      (1) Symmetrized KL divergence on transition probabilities.
      (2) A simple measure of difference between the Gaussian emissions.

    seqA, seqB: (T, B) arrays of real-valued data (e.g., your numeric scores).
    n_hidden_states: number of hidden states in the HMMs.
    n_iter: number of training iterations.

    Returns:
      float: A distance metric combining transitions + emission distributions.
    """
    hmmA = train_gaussian_hmm(seqA, n_hidden_states, n_iter)
    hmmB = train_gaussian_hmm(seqB, n_hidden_states, n_iter)

    # 1. Compare transitions
    kl_trans = sym_kl_transitions(hmmA, hmmB)

    # 2. Compare emissions (means, covars)
    #    This is just one possible approach; you can design your own measure.
    kl_emissions = sym_kl_gaussian_emissions(hmmA, hmmB)

    # Combine them. You can weight them or just sum them.
    return kl_trans + kl_emissions


def train_gaussian_hmm(seq, n_hidden_states, n_iter):
    """
    Trains a GaussianHMM on a real-valued sequence of shape (T,B).

    Returns:
      A fitted hmm.GaussianHMM model.
    """
    # seq is (T,B) => each row is an observation of dimension B
    model = hmm.GaussianHMM(n_components=n_hidden_states,
                            n_iter=n_iter,
                            covariance_type='full',
                            random_state=42,
                            verbose=False)
    model.fit(seq)  # shape (T,B)
    return model


def sym_kl_transitions(hmmA, hmmB):
    """
    Symmetrized KL divergence on transition matrices of two HMMs.
    """
    transA = hmmA.transmat_.copy()
    transB = hmmB.transmat_.copy()

    # Add epsilon to avoid log(0)
    transA += 1e-9
    transB += 1e-9

    kl_ab = np.sum(rel_entr(transA, transB))  # KL(A||B)
    kl_ba = np.sum(rel_entr(transB, transA))  # KL(B||A)
    return 0.5 * (kl_ab + kl_ba)


def sym_kl_gaussian_emissions(hmmA, hmmB):
    """
    Example measure of difference between Gaussian emissions in two HMMs.
    For each state, we compute a KL between the Gaussians, then sum or average.
    This is simplistic and does not consider mixture weights or reorderings.

    Returns:
      float: A scalar distance.
    """
    # Means: shape (n_components, n_features)
    meansA = hmmA.means_
    meansB = hmmB.means_
    covarsA = hmmA.covars_
    covarsB = hmmB.covars_

    # Ensure same number of states
    if meansA.shape != meansB.shape:
        # If the HMMs learn different numbers of states effectively,
        # you'd need a matching approach. We'll assume they match here.
        raise ValueError("HMMs have different shape of means.")

    n_states = meansA.shape[0]
    kl_total = 0.0
    for i in range(n_states):
        # KL( N(muA, SigmaA) || N(muB, SigmaB) ), symmetrized
        kl_ij = gaussian_kl(meansA[i], covarsA[i], meansB[i], covarsB[i])
        kl_ji = gaussian_kl(meansB[i], covarsB[i], meansA[i], covarsA[i])
        kl_total += 0.5 * (kl_ij + kl_ji)

    return kl_total


def gaussian_kl(mu1, Sigma1, mu2, Sigma2):
    """
    KL divergence D( N(mu1,Sigma1) || N(mu2,Sigma2) ) for full covariance Gaussians.
    Formula from standard references:
      KL = 0.5 [ log|Sigma2|/|Sigma1| - d + tr(Sigma2^-1 Sigma1)
                 + (mu2 - mu1)^T Sigma2^-1 (mu2 - mu1) ]
    """
    d = mu1.shape[0]
    # Ensure arrays
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    Sigma1 = np.asarray(Sigma1)
    Sigma2 = np.asarray(Sigma2)

    # Invert Sigma2
    invSigma2 = np.linalg.inv(Sigma2)

    # log(det(Sigma2)/det(Sigma1))
    det1 = np.linalg.det(Sigma1)
    det2 = np.linalg.det(Sigma2)
    ratio = np.log((det2 + 1e-12)/(det1 + 1e-12))

    # trace( invSigma2 * Sigma1 )
    trace_term = np.trace(invSigma2 @ Sigma1)

    # mahalanobis
    diff = (mu2 - mu1).reshape(-1, 1)
    mahal = (diff.T @ invSigma2 @ diff).item()

    kl = 0.5 * (ratio - d + trace_term + mahal)
    return kl
