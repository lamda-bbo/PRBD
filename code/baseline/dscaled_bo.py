"""

import argparse
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from torch.quasirandom import SobolEngine

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
import botorch
import math
from typing import Dict
import torch
from botorch.models import (
    FixedNoiseGP,
    SingleTaskGP,
)

from gpytorch.kernels import (
    ScaleKernel,
    MaternKernel,
    RBFKernel
)
from gpytorch.priors import (
    NormalPrior,
    GammaPrior,
    LogNormalPrior
)
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

MODELS = {
    'FixedNoiseGP': FixedNoiseGP,
    'SingleTaskGP': SingleTaskGP,
}
DIM_SCALING = {
    "default": (0.5, 0),  # scaling factor of mu and sigma in the dim
    "with_ops": (0.5, 0),
}

gp_params = {
    "const": {
        "loc": 0.0,
        "scale": 1.0
    },
    "ls": {
        "loc": 1.4,
        "scale": 1.73205
    },
    "noise": {
        "loc": -4.0,
        "scale": 1.0
    }
}


def parse_hyperparameters(gp_params: Dict[str, float], dims: int, dim_scaling: float = None):
    ls_params = gp_params.get('ls', {})
    ops_params = gp_params.get('ops', {})
    noise_params = gp_params.get('noise', {})
    if dim_scaling is not None:
        ls_params['loc'] = ls_params['loc'] + math.log(dims) * dim_scaling[0]
        # toyed with scaling the scale parameter as well
        ls_params['scale'] = (ls_params['scale'] ** 2 + math.log(dims) * dim_scaling[
            1]) ** 0.5  # Since it's std and not var, we divide by 2

    return ls_params, ops_params, noise_params


def parse_constraints(gp_constraints):
    ls_constraint = gp_constraints.get('ls', 1e-4)
    scale_constraint = gp_constraints.get('scale', 1e-4)
    noise_constraint = gp_constraints.get('noise', 1e-4)

    return ls_constraint, scale_constraint, noise_constraint


def get_covar_module(model_name, dims, gp_params: Dict = None, gp_constraints: Dict = {}):
    ls_params, ops_params, noise_params = parse_hyperparameters(
        gp_params, dims, dim_scaling=(DIM_SCALING.get(model_name)))
    ls_constraint, scale_constraint, noise_constraint = parse_constraints(
        gp_constraints)

    COVAR_MODULES = {
        'default':
            {
                'covar_module_class': RBFKernel,
                'covar_module_options': dict(
                    ard_num_dims=dims,
                    lengthscale_prior=LogNormalPrior(**ls_params),
                    lengthscale_constraint=GreaterThan(ls_constraint)
                ),
                'likelihood_class': GaussianLikelihood,
                'likelihood_options': dict(
                    noise_prior=LogNormalPrior(**noise_params),
                    noise_constraint=GreaterThan(noise_constraint)
                ),
            },

        'with_ops':
            {
                'covar_module_class': ScaleKernel,
                'covar_module_options': dict(
                    base_kernel=RBFKernel(
                        ard_num_dims=dims,
                        lengthscale_prior=LogNormalPrior(**ls_params),
                        lengthscale_constraint=GreaterThan(ls_constraint)
                    ),
                    outputscale_prior=GammaPrior(2, 0.15),
                    outputscale_constraint=GreaterThan(scale_constraint)
                ),
                'likelihood_class': GaussianLikelihood,
                'likelihood_options': dict(
                    noise_prior=LogNormalPrior(**noise_params),
                    noise_constraint=GreaterThan(noise_constraint)
                ),
            },

    }
    return COVAR_MODULES[model_name]


def build_gp_model(train_x, train_y, dims, gp_params, gp_constraints=None, model_name="default", device=None):
    if gp_constraints is None:
        gp_constraints = {}

    module_info = get_covar_module(model_name, dims, gp_params, gp_constraints)
    covar_class = module_info["covar_module_class"]
    covar_options = module_info["covar_module_options"]
    likelihood_class = module_info["likelihood_class"]
    likelihood_options = module_info["likelihood_options"]

    covar_module = covar_class(**covar_options)
    likelihood = likelihood_class(**likelihood_options)

    model = SingleTaskGP(
        train_x,
        train_y,
        covar_module=covar_module,
        likelihood=likelihood,
    )

    if device is not None:
        model = model.to(device)

    return model


def propose_rand_samples_sobol(dims, n, lb, ub, device):
    lb = torch.tensor(lb, dtype=torch.float64)
    ub = torch.tensor(ub, dtype=torch.float64)
    seed = torch.randint(int(5e5), (1,)).item()
    sobol = SobolEngine(dims, scramble=True, seed=seed)
    cands = sobol.draw(n).to(dtype=torch.float64, device=device)
    lb = lb.clone().detach().to(dtype=torch.float64, device=device)
    ub = ub.clone().detach().to(dtype=torch.float64, device=device)
    cands = cands * (ub - lb) + lb
    return cands

def optimize_acquisition(acqf, dims, lb, ub, n_points=1, n_candidates=1024, device=None):
    lb = torch.tensor(lb, dtype=torch.float64)
    ub = torch.tensor(ub, dtype=torch.float64)
    X = propose_rand_samples_sobol(dims, n_candidates, lb, ub, device=device)
    X = X.unsqueeze(1)
    with torch.no_grad():
        ei_vals = acqf(X)
    top_idx = torch.topk(ei_vals.flatten(), n_points).indices
    X_best = X[top_idx].squeeze(1)
    return X_best, ei_vals[top_idx]



def sobol_init(n, dims, lb, ub, seed, device):
    sobol = SobolEngine(dims, scramble=True, seed=seed)
    X = sobol.draw(n).to(dtype=torch.float64, device=device)
    X = lb + (ub - lb) * X
    return X


if __name__ == '__main__':
    import botorch
    import pandas as pd
    import argparse
    import random
    from benchmark import get_problem
    from utils import latin_hypercube, from_unit_cube, save_results, save_args

    parser = argparse.ArgumentParser()
    parser.add_argument('--func', default='hartmann6_50', type=str)
    parser.add_argument('--max_samples', default=600, type=int)
    parser.add_argument('--init_samples', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--root_dir', default='synthetic_logs', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()
    print(args)

    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    botorch.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    save_config = {
        'save_interval': 20,
        'root_dir': 'logs/' + args.root_dir,
        'algo': 'D_scaled_bo',
        'func': args.func,
        'seed': args.seed
    }
    func = get_problem(args.func, save_config)

    save_args(
        'config/' + args.root_dir,
        'D_scaled_bo',
        args.func,
        args.seed,
        args
    )

    dims, lb, ub = func.dims, func.lb, func.ub
    points = latin_hypercube(args.init_samples, dims)
    points = from_unit_cube(points, lb, ub)
    train_x, train_y = [], []
    for i in range(args.init_samples):
        y = func(points[i])
        train_x.append(points[i])
        train_y.append(y)

    train_x = torch.tensor(train_x, dtype=torch.float64, device=device)
    train_y = torch.tensor(train_y, dtype=torch.float64, device=device).unsqueeze(-1)
    sample_counter = args.init_samples
    #best_y = [(sample_counter, np.max(train_y))]
    best_y = [(sample_counter, train_y.max().item())]

    while True:
        model = build_gp_model(
            train_x,
            train_y,
            dims,
            gp_params,
            device=device
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll, options={"maxiter": 50})
        acqf = ExpectedImprovement(
            model=model,
            best_f=train_y.max().item(),
        )

        proposed_X, _ = optimize_acquisition(acqf, dims, lb, ub, n_points=1, n_candidates=2048, device=device)
        proposed_Y = [func(X.detach().cpu().numpy()) for X in proposed_X]
        proposed_Y = torch.tensor(proposed_Y, dtype=torch.float64, device=device).unsqueeze(-1)

        train_x = torch.cat([train_x, proposed_X], dim=0)
        train_y = torch.cat([train_y, proposed_Y], dim=0)
        sample_counter += len(proposed_X)
        best_y.append((sample_counter, train_y.max().item()))
        if sample_counter >= args.max_samples:
            break
        del model
        del mll
        del acqf
        gc.collect()
        torch.cuda.empty_cache()

    print('best f(x):', best_y[-1][1])
"""

import os
NUM_THREADS = 6
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
import argparse
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from torch.quasirandom import SobolEngine

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
import botorch
import math
from typing import Dict
import torch
from botorch.models import (
    FixedNoiseGP,
    SingleTaskGP,
)

from gpytorch.kernels import (
    ScaleKernel,
    MaternKernel,
    RBFKernel
)
from gpytorch.priors import (
    NormalPrior,
    GammaPrior,
    LogNormalPrior
)
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

MODELS = {
    'FixedNoiseGP': FixedNoiseGP,
    'SingleTaskGP': SingleTaskGP,
}
DIM_SCALING = {
    "default": (0.5, 0),  # scaling factor of mu and sigma in the dim
    "with_ops": (0.5, 0),
}

gp_params = {
    "const": {
        "loc": 0.0,
        "scale": 1.0
    },
    "ls": {
        "loc": 1.4,
        "scale": 1.73205
    },
    "noise": {
        "loc": -4.0,
        "scale": 1.0
    }
}


def parse_hyperparameters(gp_params: Dict[str, float], dims: int, dim_scaling: float = None):
    ls_params = gp_params.get('ls', {})
    ops_params = gp_params.get('ops', {})
    noise_params = gp_params.get('noise', {})
    if dim_scaling is not None:
        ls_params['loc'] = ls_params['loc'] + math.log(dims) * dim_scaling[0]
        # toyed with scaling the scale parameter as well
        ls_params['scale'] = (ls_params['scale'] ** 2 + math.log(dims) * dim_scaling[
            1]) ** 0.5  # Since it's std and not var, we divide by 2

    return ls_params, ops_params, noise_params


def parse_constraints(gp_constraints):
    ls_constraint = gp_constraints.get('ls', 1e-4)
    scale_constraint = gp_constraints.get('scale', 1e-4)
    noise_constraint = gp_constraints.get('noise', 1e-4)

    return ls_constraint, scale_constraint, noise_constraint


def get_covar_module(model_name, dims, gp_params: Dict = None, gp_constraints: Dict = {}):
    ls_params, ops_params, noise_params = parse_hyperparameters(
        gp_params, dims, dim_scaling=(DIM_SCALING.get(model_name)))
    ls_constraint, scale_constraint, noise_constraint = parse_constraints(
        gp_constraints)

    COVAR_MODULES = {
        'default':
            {
                'covar_module_class': RBFKernel,
                'covar_module_options': dict(
                    ard_num_dims=dims,
                    lengthscale_prior=LogNormalPrior(**ls_params),
                    lengthscale_constraint=GreaterThan(ls_constraint)
                ),
                'likelihood_class': GaussianLikelihood,
                'likelihood_options': dict(
                    noise_prior=LogNormalPrior(**noise_params),
                    noise_constraint=GreaterThan(noise_constraint)
                ),
            },

        'with_ops':
            {
                'covar_module_class': ScaleKernel,
                'covar_module_options': dict(
                    base_kernel=RBFKernel(
                        ard_num_dims=dims,
                        lengthscale_prior=LogNormalPrior(**ls_params),
                        lengthscale_constraint=GreaterThan(ls_constraint)
                    ),
                    outputscale_prior=GammaPrior(2, 0.15),
                    outputscale_constraint=GreaterThan(scale_constraint)
                ),
                'likelihood_class': GaussianLikelihood,
                'likelihood_options': dict(
                    noise_prior=LogNormalPrior(**noise_params),
                    noise_constraint=GreaterThan(noise_constraint)
                ),
            },

    }
    return COVAR_MODULES[model_name]


def build_gp_model(train_x, train_y, dims, gp_params, gp_constraints=None, model_name="default", device=None):
    if gp_constraints is None:
        gp_constraints = {}

    module_info = get_covar_module(model_name, dims, gp_params, gp_constraints)
    covar_class = module_info["covar_module_class"]
    covar_options = module_info["covar_module_options"]
    likelihood_class = module_info["likelihood_class"]
    likelihood_options = module_info["likelihood_options"]

    covar_module = covar_class(**covar_options)
    likelihood = likelihood_class(**likelihood_options)

    model = SingleTaskGP(
        train_x,
        train_y,
        covar_module=covar_module,
        likelihood=likelihood,
    )

    if device is not None:
        model = model.to(device)

    return model


def propose_rand_samples_sobol(dims, n, lb, ub, device):
    lb = torch.tensor(lb, dtype=torch.float32)
    ub = torch.tensor(ub, dtype=torch.float32)
    seed = torch.randint(int(5e5), (1,)).item()
    sobol = SobolEngine(dims, scramble=True, seed=seed)
    cands = sobol.draw(n).to(dtype=torch.float32, device=device)
    lb = lb.clone().detach().to(dtype=torch.float32, device=device)
    ub = ub.clone().detach().to(dtype=torch.float32, device=device)
    cands = cands * (ub - lb) + lb
    return cands

def optimize_acquisition(acqf, dims, lb, ub, n_points=1, n_candidates=1024, device=None):
    lb = torch.tensor(lb, dtype=torch.float32)
    ub = torch.tensor(ub, dtype=torch.float32)
    X = propose_rand_samples_sobol(dims, n_candidates, lb, ub, device=device)
    X = X.unsqueeze(1)
    with torch.no_grad():
        ei_vals = acqf(X)
    top_idx = torch.topk(ei_vals.flatten(), n_points).indices
    X_best = X[top_idx].squeeze(1)
    return X_best, ei_vals[top_idx]



def sobol_init(n, dims, lb, ub, seed, device):
    sobol = SobolEngine(dims, scramble=True, seed=seed)
    X = sobol.draw(n).to(dtype=torch.float32, device=device)
    X = lb + (ub - lb) * X
    return X


if __name__ == '__main__':
    import botorch
    import pandas as pd
    import gc
    import argparse
    import random
    from benchmark import get_problem
    from utils import latin_hypercube, from_unit_cube, save_results, save_args

    parser = argparse.ArgumentParser()
    parser.add_argument('--func', default='hartmann6_50', type=str)
    parser.add_argument('--max_samples', default=600, type=int)
    parser.add_argument('--init_samples', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--root_dir', default='synthetic_logs', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()
    print(args)

    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    botorch.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    save_config = {
        'save_interval': 20,
        'root_dir': 'logs/' + args.root_dir,
        'algo': 'D_scaled_bo',
        'func': args.func,
        'seed': args.seed
    }
    func = get_problem(args.func, save_config)

    save_args(
        'config/' + args.root_dir,
        'D_scaled_bo',
        args.func,
        args.seed,
        args
    )

    dims, lb, ub = func.dims, func.lb, func.ub
    points = latin_hypercube(args.init_samples, dims)
    points = from_unit_cube(points, lb, ub)
    train_x, train_y = [], []
    for i in range(args.init_samples):
        y = func(points[i])
        train_x.append(points[i])
        train_y.append(y)

    train_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_y, dtype=torch.float32, device=device).unsqueeze(-1)
    sample_counter = args.init_samples
    #best_y = [(sample_counter, np.max(train_y))]
    best_y = [(sample_counter, train_y.max().item())]

    while True:
        model = build_gp_model(
            train_x,
            train_y,
            dims,
            gp_params,
            device=device
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll, options={"maxiter": 50})
        acqf = ExpectedImprovement(
            model=model,
            best_f=train_y.max().item(),
        )

        proposed_X, _ = optimize_acquisition(acqf, dims, lb, ub, n_points=1, n_candidates=2048, device=device)
        proposed_Y = [func(X.detach().cpu().numpy()) for X in proposed_X]
        proposed_Y = torch.tensor(proposed_Y, dtype=torch.float32, device=device).unsqueeze(-1)

        train_x = torch.cat([train_x, proposed_X], dim=0)
        train_y = torch.cat([train_y, proposed_Y], dim=0)
        sample_counter += len(proposed_X)
        best_y.append((sample_counter, train_y.max().item()))
        if sample_counter >= args.max_samples:
            break
        del model
        del mll
        del acqf
        gc.collect()
        torch.cuda.empty_cache()

    print('best f(x):', best_y[-1][1])