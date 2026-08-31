import numpy as np
from abc import ABC, abstractmethod
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,Matern
from sklearn.linear_model import LinearRegression

class InteractionMetric(ABC):
    def __init__(self):
        self.E = None

    @abstractmethod
    def compute(self, *args):
        pass

    def normalize(self, E):
        E = np.array(E, dtype=float)
        E = (E - np.min(E)) / (np.max(E) - np.min(E) + 1e-8)
        return E


class AverageImprovementInteraction(InteractionMetric):
    def __init__(self, dim ,device=None):
        super().__init__()
        self.dim = dim
        self.prev_y = None

    def update(self, groups, ema_Y):
        '''
        assume max problem
        '''

    def compute(self, t, groups, interaction_sum, interaction_count, prior_mean=0.5):
        #self.update(groups, ema_Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_interaction = np.divide(
                interaction_sum,
                interaction_count,
                out=np.zeros_like(interaction_sum),
                where=interaction_count != 0
            )

        avg_interaction_raw = avg_interaction
        avg_interaction_norm = avg_interaction_raw.copy()
        mask = interaction_count > 0
        if np.any(mask):
            v = avg_interaction_norm[mask]
            v_min = v.min()
            v_ptp = np.ptp(v)
            if v_ptp > 0:
                avg_interaction_norm[mask] = (v - v_min) / (v_ptp + 1e-8)
            else:
                avg_interaction_norm[mask] = 0.0
        confidence = np.tanh(interaction_count / 5.0)
        E = prior_mean * (1 - confidence) + avg_interaction_norm * confidence
        d = E.shape[0]
        epsilon = 1e-12
        jitter = np.random.uniform(0, epsilon, size=(d, d))
        jitter = np.triu(jitter, k=1)
        jitter = (jitter + jitter.T) / 2
        np.fill_diagonal(jitter, 0.0)
        E += jitter
        E = self.normalize(E)
        return E



import torch
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, RBFKernel,AdditiveKernel
from gpytorch.mlls import ExactMarginalLogLikelihood


class MarginalLikelihoodInteraction:
    def __init__(self, dim, device="cpu", ard=True, optimize_l=False):
        self.device = device or torch.device("cpu")
        self.dim = dim
        self.ard = ard
        self.optimize_l = optimize_l

    def _build_gp(self, X, y, kernel=None):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)

        if kernel is None:
            kernel = ScaleKernel(RBFKernel(lengthscale=0.5,ard_num_dims=X.shape[1] if self.ard else None))

        gp = SingleTaskGP(X, y.unsqueeze(-1), covar_module=kernel)

        if self.optimize_l:
            gp.train()
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
            for _ in range(50):
                optimizer.zero_grad()
                loss = -mll(gp(X), y)
                loss.backward()
                optimizer.step()
            gp.eval()

        return gp

    def compute(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.X = torch.tensor(self.X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(self.y, dtype=torch.float32, device=self.device).view(-1)
        d = self.X.shape[1]
        E = torch.zeros(d, d, device=self.device)

        for i in range(d):
            for j in range(i + 1, d):
                X_ij = self.X[:, [i, j]]
                kernels = [
                    ScaleKernel(RBFKernel(lengthscale=0.5, active_dims=[0])),
                    ScaleKernel(RBFKernel(lengthscale=0.5,active_dims=[1]))
                ]
                additive_kernel = AdditiveKernel(*kernels)
                gp = self._build_gp(X_ij, self.y, kernel=additive_kernel)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                log_ml_additive = mll(gp(X_ij), self.y).sum().item()
                del gp

                gp_joint = self._build_gp(X_ij, self.y)
                mll_joint = ExactMarginalLogLikelihood(gp_joint.likelihood, gp_joint)
                log_ml_joint = mll_joint(gp_joint(X_ij), self.y).sum().item()
                del gp_joint
                E[i, j] = E[j, i] = log_ml_joint - log_ml_additive

        E=E.cpu().numpy()
        d = E.shape[0]
        epsilon = 1e-12
        jitter = np.random.uniform(0, epsilon, size=(d, d))
        jitter = np.triu(jitter, k=1)
        jitter = (jitter + jitter.T) / 2
        np.fill_diagonal(jitter, 0.0)
        E += jitter

        return E

class RegressionInteractionScore(InteractionMetric):
    def __init__(self, dim, device=None,interaction_order=2):
        super().__init__()
        self.order = interaction_order

    def compute(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.d = self.X.shape[1]
        n_samples, d = self.X.shape
        E = np.zeros((d, d))
        X_design = self.X.copy()
        inter_idx = []
        for i in range(d):
            for j in range(i+1, d):
                xi_xj = (self.X[:, i] * self.X[:, j]).reshape(-1, 1)
                X_design = np.hstack((X_design, xi_xj))
                inter_idx.append((i, j))

        model = LinearRegression()
        model.fit(X_design, self.y)
        coefs = model.coef_
        for idx, (i, j) in enumerate(inter_idx):
            E[i, j] = E[j, i] = abs(coefs[d + idx])
        del model

        d = E.shape[0]
        epsilon = 1e-12
        jitter = np.random.uniform(0, epsilon, size=(d, d))
        jitter = np.triu(jitter, k=1)
        jitter = (jitter + jitter.T)/2
        np.fill_diagonal(jitter, 0.0)
        E += jitter
        E = self.normalize(E)
        return E