import time
import warnings
from copy import deepcopy

import numpy as np
import numpyro

from numpyro.util import enable_x64
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import qmc


def run_saasbo_one_epoch(
    X,
    Y,
    dims,
    batch_size,
    f,
    lb,
    ub,
    seed=None,
    alpha=0.1,
    num_warmup=512,
    num_samples=256,
    thinning=16,
    kernel="rbf",
    device="cpu",
):
    """
    Run SAASBO and approximately minimize f.
    Arguments:
    f: function to minimize. should accept a D-dimensional np.array as argument. the input domain of f
        is assumed to be the D-dimensional rectangular box bounded by lower and upper bounds lb and ub.
    lb: D-dimensional vector of lower bounds (np.array)
    ub: D-dimensional vector of upper bounds (np.array)
    max_evals: The total evaluation budget
    num_init_evals: The initial num_init_evals query points are chosen at random from the input
        domain using a Sobol sequence. must satisfy num_init_evals < max_evals.
    seed: Random number seed (int or None); defaults to None
    alpha: Positive float that controls the level of sparsity (smaller alpha => more sparsity).
        defaults to alpha = 0.1.
    num_warmup: The number of warmup samples to use in HMC inference. defaults to 512.
    num_samples: The number of post-warmup samples to use in HMC inference. defaults to 256.
    thinning: Positive integer that controls the fraction of posterior hyperparameter samples
        that are used to compute the expected improvement. for example thinning==2 will use every
        other sample. defaults to no thinning (thinning==1).
    num_restarts_ei: The number of restarts for L-BFGS-B when optimizing EI.
    kernel: By default saasbo uses rbf, but matern is also supported.
    device: Whether to use cpu or gpu. defaults to "cpu".
    Returns:
        X: np.array containing all query points (of which there are max_evals many)
        Y: np.array containing all observed function evaluations (of which there are max_evals many)
    """
    # if max_evals <= num_init_evals:
    #     raise ValueError("Must choose max_evals > num_init_evals.")
    x_next=None
    return x_next
