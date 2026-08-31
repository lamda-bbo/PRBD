import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube
from .turbo_1 import Turbo1

import math
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel, AdditiveKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP


# GP Model
class add_GP(ExactGP):
    def __init__(self, groups, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, use_ard):
        super(add_GP, self).__init__(train_x, train_y, likelihood)
        self.groups = groups
        self.use_ard = use_ard
        self.mean_module = ConstantMean()
        kernels = []
        for g in self.groups:
            ard_dims = len(g) if self.use_ard else None
            k = ScaleKernel(
                MaternKernel(nu=2.5, active_dims=g,
                             lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims),
                             outputscale_constraint=outputscale_constraint
            )
            kernels.append(k)
        base_kernel = AdditiveKernel(*kernels)
        self.covar_module = base_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)



def add_train_gp(groups, train_x, train_y, use_ard, num_steps, hypers={}):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # Create hyper parameter bounds
    noise_constraint = Interval(5e-4, 0.2)
    if use_ard:
        lengthscale_constraint = Interval(0.005, 2.0)
    else:
        lengthscale_constraint = Interval(0.005, math.sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    model = add_GP(
        groups = groups,
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        use_ard=use_ard,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        #hypers["covar_module.outputscale"] = 1.0
        #hypers["covar_module.base_kernel.lengthscale"] = 0.5
        for i, k in enumerate(model.covar_module.kernels):
            hypers[f"covar_module.kernels.{i}.outputscale"] = 1.0
            hypers[f"covar_module.kernels.{i}.base_kernel.lengthscale"] = 0.5
        hypers["likelihood.noise"] = 0.005
        model.initialize(**hypers)

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model



class Turbo1_Component(Turbo1):
    def _create_candidates(self, groups, X, fX, length, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = add_train_gp(
                groups=groups, train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        """
        weights_list = []
        for k in gp.covar_module.kernels:  # 遍历每个子核
            # k 是 ScaleKernel(MaternKernel)
            ls = k.base_kernel.lengthscale  # 取内部 MaternKernel 的 lengthscale
            weights_list.append(ls.cpu().detach().numpy().ravel())

        weights = np.concatenate(weights_list)  # 拼成全维度向量
        """

        weights = np.zeros(X.shape[1])  # 初始化每个维度的 lengthscale

        for k in gp.covar_module.kernels:  # 遍历每个子核
            ls = k.base_kernel.lengthscale.detach().cpu().numpy().ravel()
            for i, dim in enumerate(k.active_dims):
                weights[dim] = ls[i]  # 直接映射，保证每个维度对应正确

        #weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def optimize(self, groups, X_init=None, fX_init=None, n=1, precision_record = False):
        print(groups)
        fX_init = fX_init.reshape(-1, 1)
        cnt = 0
        self.n_evals = 0
        X_sample, Y_sample = [], []
        while self.n_evals < self.max_evals and cnt < n:
            cnt += 1
            # Initialize parameters
            self._restart()

            # Update budget and set as initial data for this TR
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)


            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

                # Standardize values
                fX = deepcopy(self._fX).ravel()

                # Create th next batch
                X_cand, y_cand, _ = self._create_candidates(
                    groups, X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                )
                X_next = self._select_candidates(X_cand, y_cand)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)

                ## Evaluate batch
                #fX_next = np.array([[self.f(x)] for x in X_next])

                # Evaluate batch
                if precision_record == True:
                    fX_next = np.array([[self.f(x, cur_group=groups)] for x in X_next])
                else:
                    fX_next = np.array([[self.f(x)] for x in X_next])

                for i in range(len(X_next)):
                    X_sample.append(X_next[i])
                    Y_sample.append(fX_next[i, 0])

                # Update trust region
                self._adjust_length(fX_next)

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))

        return X_sample, Y_sample


class Turbo1_VS_Component(Turbo1):
    def optimize(self, X_init, fX_init, feature_idx, uipt_solver, n=1):
        fX_init = fX_init.reshape(-1, 1)
        cnt = 0
        self.n_evals = 0
        X_sample, Y_sample = [], []
        while self.n_evals < self.max_evals and cnt < n:
            cnt += 1
            # Initialize parameters
            self._restart()
            
            # Update budget and set as initial data for this TR
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

                # Standardize values
                fX = deepcopy(self._fX).ravel()

                # Create th next batch
                X_cand, y_cand, _ = self._create_candidates(
                    X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                )
                X_next = self._select_candidates(X_cand, y_cand)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)
                
                # Evaluate batch
                fX_next = []
                for i in range(len(X_next)):
                    fixed_variables = {idx: float(v) for idx, v in zip(feature_idx, X_next[i])}
                    new_x = uipt_solver.get_full_variable(
                        fixed_variables, 
                        self.lb, 
                        self.ub
                    )
                    value = self.f(new_x)
                    fX_next.append([value])
                    
                    # update global store
                    X_sample.append(new_x)
                    Y_sample.append(value)
                    uipt_solver.update(new_x, -value)

                # Update trust region
                self._adjust_length(fX_next)

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

        return X_sample, Y_sample
