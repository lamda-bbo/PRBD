import os
NUM_THREADS = 6
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf as botorch_optimize_acqf
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, AdditiveKernel
from botorch.utils.sampling import draw_sobol_samples
from torch.quasirandom import SobolEngine
import math

from IndexAdditiveBO.Grouping import group_sampler
from IndexAdditiveBO.interaction_metric import AverageImprovementInteraction,MarginalLikelihoodInteraction,RegressionInteractionScore

from utils import latin_hypercube, from_unit_cube
import networkx as nx
import random
from itertools import product
from disjoint_set import DisjointSet
from baseline.dscaled_bo import build_gp_model, optimize_acquisition, gp_params
from inner_optimizer import Turbo1_Component

class IndexAdditiveBO:
    def __init__(self, func, dims, lb, ub, true_groups,
                 max_group_size=5, sample_init_size= 10, sample_batch_size=3,
                 acqf_name = 'ei', kernel_type='Matern', acq_opt_mode = 'group',
                 acq_opt_method = 'sobol', group_update_interval=1,
                 metric_type="random" , metric_classes=("AverageImprovementInteraction","MarginalLikelihoodInteraction","RegressionInteractionScore"),
                 grouping_strategy= "Kmeans",weights=None ,weight_strategy="Learn" , model = None,model_path = None, precision_record = True, device = "cpu"):
        assert len(lb) == dims and len(ub) == dims
        self.func = func
        self.dims = dims
        self.tdtype = torch.float32
        self.lb = torch.tensor(lb, dtype=self.tdtype)
        self.ub = torch.tensor(ub, dtype=self.tdtype)
        self.true_groups = true_groups
        self.sample_init_size = sample_init_size
        self.sample_batch_size = sample_batch_size
        self.max_group_size = max_group_size
        self.group_update_interval = group_update_interval
        self.samples = []
        self.ema_Y = []
        self.ema_alpha = 1.0
        self.curt_best_sample = None
        self.curt_best_value = float('-inf')
        self.best_value_trace = []
        self.value_trace = []
        self.sample_counter = 0
        self.groups = None
        self.t = 0

        self.kernel_type = kernel_type
        self.acqf_name = acqf_name
        self.acq_opt_mode = acq_opt_mode
        self.acq_opt_method = acq_opt_method

        self.metric_type = metric_type
        self.metric_classes = metric_classes
        self.grouping_strategy = grouping_strategy
        self.weights = weights
        self.weight_strategy = weight_strategy
        self.model = None
        self.precision_record = precision_record
        self.device = device
        if self.metric_classes == "AverageImprovementInteraction":
            self.interaction_sum = np.zeros((dims, dims))
            self.interaction_count = np.zeros((dims, dims))

        self.init_train()
        self.init_group()


    def init_group(self):
        if self.grouping_strategy in ("true","random"):
            return
        metric_map = {
            "AverageImprovementInteraction": AverageImprovementInteraction,
            "MarginalLikelihoodInteraction": MarginalLikelihoodInteraction,
            "RegressionInteractionScore": RegressionInteractionScore
        }
        if self.metric_type == "single":
            self.index_obj = metric_map[self.metric_classes](self.dims,device=self.device)
        else:
            raise ValueError(f"unknow metric_type: {self.metric_type}")

    def propose_rand_samples_sobol(self, dims, n, lb, ub):
        seed = torch.randint(int(5e5), (1,)).item()
        sobol = SobolEngine(dims, scramble=True, seed=seed)
        cands = sobol.draw(n).to(dtype=self.tdtype, device=self.device)
        lb = lb.clone().detach().to(dtype=self.tdtype, device=self.device)
        ub = ub.clone().detach().to(dtype=self.tdtype, device=self.device)
        cands = cands * (ub - lb) + lb
        return cands

    def get_groups(self):
        if self.metric_type == "single":
            if self.metric_classes == "AverageImprovementInteraction":
                if self.t == 0:
                    perm = np.random.permutation(self.dims)
                    groups = []
                    for i in range(0, self.dims, self.max_group_size):
                        groups.append(list(perm[i:i + self.max_group_size]))
                    return groups
                else:
                    E = self.index_obj.compute(self.t, self.groups, self.interaction_sum,self.interaction_count)
            elif self.metric_classes in ["SobolInteraction","MarginalLikelihoodInteraction","RegressionInteractionScore"]:
                X, y = list(zip(*self.samples))
                E = self.index_obj.compute(X, y)
        else:
            pass
        return group_sampler(self.grouping_strategy, E, n_groups = math.ceil(self.dims / self.max_group_size), max_group_size=self.max_group_size)

    def ema(self, y_new):
        if len(self.ema_Y) == 0:
            ema_value = y_new
        else:
            ema_value = self.ema_alpha * y_new + (1 - self.ema_alpha) * self.ema_Y[-1]
        self.ema_Y.append(ema_value)
        return self.ema_Y

    def init_train(self):
        assert len(self.samples) == 0
        points = latin_hypercube(self.sample_init_size, self.dims)
        points = from_unit_cube(points, self.lb.numpy(), self.ub.numpy())
        for i in range(self.sample_init_size):
            if self.precision_record:
                y = self.func(points[i], cur_group = 1)
            else:
                y = self.func(points[i])
            self.ema(y)
            self.samples.append((points[i], y))

        # update current best sample information
        self.sample_counter += len(self.samples)
        X_sample, Y_sample = zip(*self.samples)
        best_sample_idx = np.argmax(Y_sample)
        self.curt_best_sample, self.curt_best_value = self.samples[best_sample_idx]
        self.best_value_trace.append((self.sample_counter, self.curt_best_value))
        print('=' * 10)
        print('sample_batch_size: {}'.format(self.sample_batch_size))
        print('collect {} samples for initializing MCTS'.format(len(self.samples)))
        print('dims: {}'.format(self.dims))
        print('=' * 10)

    def update_groups(self):
        if self.grouping_strategy == "true":
            self.groups = self.true_groups
        elif self.grouping_strategy == "random":
            perm = np.random.permutation(self.dims)
            self.groups = []
            for i in range(0, self.dims, self.max_group_size):
                self.groups.append(list(perm[i:i + self.max_group_size]))
        elif self.grouping_strategy == "RTree":
            graph = nx.empty_graph(self.dims)
            disjoint_set = DisjointSet()
            connections_made = 0
            while connections_made < max(1, int(0.2 * self.dims)):
                edge_in = random.randint(0, self.dims - 1)
                edge_out = random.randint(0, self.dims - 1)
                if edge_in == edge_out or disjoint_set.connected(edge_out, edge_in):
                    continue
                graph.add_edge(edge_in, edge_out)
                disjoint_set.union(edge_in, edge_out)
                connections_made += 1
            groups = [tuple([v]) for v in nx.isolates(graph)]
            groups += [tuple(sorted(e)) for e in graph.edges()]
            self.groups = groups
        elif self.grouping_strategy == "Tree":
            X_train, Y_train = zip(*self.samples)
            X_train = torch.from_numpy(np.array(X_train)).to(dtype=self.tdtype).to(self.device)
            Y_train = torch.tensor(np.array(Y_train, dtype=np.float64)).unsqueeze(-1).to(dtype=self.tdtype).to(self.device)
            if self.t == 0:
                graph = nx.empty_graph(self.dims)
                disjoint_set = DisjointSet()
                connections_made = 0
                while connections_made < max(1, int(0.2 * self.dims)):
                    edge_in = random.randint(0, self.dims - 1)
                    edge_out = random.randint(0, self.dims - 1)
                    if edge_in == edge_out or disjoint_set.connected(edge_out, edge_in):
                        continue
                    graph.add_edge(edge_in, edge_out)
                    disjoint_set.union(edge_in, edge_out)
                    connections_made += 1
                self.groups = [tuple([v]) for v in nx.isolates(graph)]
                self.groups += [tuple(sorted(e)) for e in graph.edges()]
                kernels = []
                for g in self.groups:
                    if self.kernel_type == 'RBF':
                        k = ScaleKernel(
                            RBFKernel(active_dims=torch.tensor(g, dtype=torch.long),
                                      lengthscale=torch.tensor([np.sqrt(len(g)) / 10.0],
                                                               dtype=self.tdtype))
                        )
                    elif self.kernel_type == 'Matern':
                        k = ScaleKernel(
                            MaternKernel(nu=2.5, active_dims=torch.tensor(g, dtype=torch.long),
                                         lengthscale=torch.tensor([np.sqrt(len(g)) / 10.0],
                                                                  dtype=self.tdtype))
                        )
                    else:
                        raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
                    kernels.append(k)

                add_kernel = AdditiveKernel(*kernels).to(self.device).to(dtype=self.tdtype)
                gp_model = SingleTaskGP(X_train, Y_train, covar_module=add_kernel).to(self.device).to(
                    dtype=self.tdtype)
                mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
                fit_gpytorch_model(mll, options={"maxiter": 50})

                with torch.no_grad():
                    self.cur_likelihood = mll(gp_model(X_train), Y_train).sum().item()
            else:
                dim = self.dims
                graph = nx.Graph()
                graph.add_nodes_from(range(dim))
                for g in self.groups:
                    if len(g) == 2:
                        graph.add_edge(*g)

                disjoint_set = DisjointSet()
                for i, j in graph.edges():
                    disjoint_set.union(i, j)

                hypotheses_set = set()
                frozen_groups = frozenset(tuple(g) for g in self.groups)
                hypotheses_set.add(frozen_groups)

                all_edges = [(i, j) for i in range(dim) for j in range(i)]
                max_iter = 20
                while len(hypotheses_set) < max_iter:
                    if len(graph.edges()) < dim - 1:
                        np.random.shuffle(all_edges)
                        for i, j in all_edges:
                            if not disjoint_set.connected(i, j):
                                graph_copy = graph.copy()
                                disjoint_set_copy = DisjointSet()
                                for u, v in graph_copy.edges():
                                    disjoint_set_copy.union(u, v)

                                graph_copy.add_edge(i, j)
                                disjoint_set_copy.union(i, j)

                                new_groups = [tuple([v]) for v in nx.isolates(graph_copy)]
                                new_groups += [tuple(sorted(e)) for e in graph_copy.edges()]
                                frozen_groups = frozenset(tuple(g) for g in new_groups)

                                if frozen_groups not in hypotheses_set:
                                    hypotheses_set.add(frozen_groups)

                                    kernels = []
                                    for g in new_groups:
                                        if self.kernel_type == 'RBF':
                                            k = ScaleKernel(
                                                RBFKernel(active_dims=torch.tensor(g, dtype=torch.long),
                                                          lengthscale=torch.tensor([np.sqrt(len(g)) / 10.0],
                                                                                   dtype=self.tdtype))
                                            )
                                        elif self.kernel_type == 'Matern':
                                            k = ScaleKernel(
                                                MaternKernel(nu=2.5, active_dims=torch.tensor(g, dtype=torch.long),
                                                             lengthscale=torch.tensor([np.sqrt(len(g)) / 10.0],
                                                                                      dtype=self.tdtype))
                                            )
                                        else:
                                            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
                                        kernels.append(k)

                                    add_kernel = AdditiveKernel(*kernels).to(self.device).to(dtype=self.tdtype)
                                    gp_model = SingleTaskGP(X_train, Y_train, covar_module=add_kernel).to(self.device).to(
                                        dtype=self.tdtype)
                                    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
                                    fit_gpytorch_model(mll, options={"maxiter": 50})

                                    with torch.no_grad():
                                        val = mll(gp_model(X_train), Y_train).sum().item()

                                    if val > self.cur_likelihood:
                                        self.groups = new_groups
                                        self.cur_likelihood = val
                                        graph = graph_copy
                                        disjoint_set = disjoint_set_copy

                                if len(hypotheses_set) >= max_iter:
                                    break

                    else:
                        edges = list(graph.edges())
                        iter_candidates = 10
                        for _ in range(iter_candidates):
                            if not edges:
                                break

                            i_del, j_del = edges[np.random.choice(len(edges))]
                            graph_copy = graph.copy()
                            graph_copy.remove_edge(i_del, j_del)

                            comp_1 = list(nx.node_connected_component(graph_copy, i_del))
                            comp_2 = list(nx.node_connected_component(graph_copy, j_del))

                            candidate_edges = set(product(comp_1, comp_2))
                            if len(candidate_edges) > 1 and (i_del, j_del) in candidate_edges:
                                candidate_edges.remove((i_del, j_del))
                            candidate_edges = list(candidate_edges)

                            if not candidate_edges:
                                continue

                            i_add, j_add = candidate_edges[np.random.choice(len(candidate_edges))]
                            graph_copy.add_edge(i_add, j_add)

                            new_groups = [tuple([v]) for v in nx.isolates(graph_copy)]
                            new_groups += [tuple(sorted(e)) for e in graph_copy.edges()]
                            frozen_groups = frozenset(tuple(g) for g in new_groups)

                            if frozen_groups not in hypotheses_set:
                                hypotheses_set.add(frozen_groups)
                                kernels = []
                                for g in new_groups:
                                    if self.kernel_type == 'RBF':
                                        k = ScaleKernel(
                                            RBFKernel(active_dims=torch.tensor(g, dtype=torch.long),
                                                      lengthscale=torch.tensor([np.sqrt(len(g)) / 10.0],
                                                                               dtype=self.tdtype))
                                        )
                                    elif self.kernel_type == 'Matern':
                                        k = ScaleKernel(
                                            MaternKernel(nu=2.5, active_dims=torch.tensor(g, dtype=torch.long),
                                                         lengthscale=torch.tensor([np.sqrt(len(g)) / 10.0],
                                                                                  dtype=self.tdtype))
                                        )
                                    else:
                                        raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
                                    kernels.append(k)

                                add_kernel = AdditiveKernel(*kernels).to(self.device).to(dtype=self.tdtype)
                                gp_model = SingleTaskGP(X_train, Y_train, covar_module=add_kernel).to(self.device).to(
                                    dtype=self.tdtype)
                                mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
                                fit_gpytorch_model(mll, options={"maxiter": 50})

                                with torch.no_grad():
                                    val = mll(gp_model(X_train), Y_train).sum().item()

                                if val > self.cur_likelihood:
                                    self.groups = new_groups
                                    self.cur_likelihood = val
                                    graph = graph_copy

                            if len(hypotheses_set) >= max_iter:
                                break

        else:
            self.groups = self.get_groups()

    def fit_gp(self):
        self.update_groups()

        X_train, Y_train = zip(*self.samples)
        X_train = torch.from_numpy(np.array(X_train)).to(dtype=self.tdtype).to(self.device)
        Y_train = torch.tensor(np.array(Y_train, dtype=np.float64)).unsqueeze(-1).to(dtype=self.tdtype).to(self.device)
        # 构建加性核
        kernels = []
        for g in self.groups:
            if self.kernel_type == 'RBF':
                k = ScaleKernel(
                    RBFKernel(active_dims=torch.tensor(g, dtype=torch.long),
                              lengthscale=torch.tensor([np.sqrt(len(g)) / 10.0], dtype=self.tdtype))
                )
            elif self.kernel_type == 'Matern':
                k = ScaleKernel(
                    MaternKernel(nu=2.5, active_dims=torch.tensor(g, dtype=torch.long),
                                 lengthscale=torch.tensor([np.sqrt(len(g)) / 10.0], dtype=self.tdtype))
                )
            else:
                raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
            kernels.append(k)

        add_kernel = AdditiveKernel(*kernels).to(self.device).to(dtype=self.tdtype)
        self.gp_model = SingleTaskGP(X_train, Y_train, covar_module=add_kernel).to(self.device).to(dtype=self.tdtype)
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        fit_gpytorch_model(mll, options={"maxiter": 50})

    def optimize_acqf_sobol(self, acq_function, gp_model, best_f, lb, ub, n_points=1, n_candidates=1024):
        device = next(gp_model.parameters()).device
        X = self.propose_rand_samples_sobol(self.dims, n_candidates, lb, ub)
        X = X.unsqueeze(1)

        acqf = acq_function
        with torch.no_grad():
            ei_vals = acqf(X)

        top_idx = torch.topk(ei_vals.flatten(), n_points).indices
        X_best = X[top_idx].squeeze(1)
        return X_best, ei_vals[top_idx]

    def collect_samples(self):
        train_x, train_y = zip(*self.samples)
        np_train_x = np.vstack(train_x)
        np_train_y  = np.array(train_y)
        new_samples = []
        if self.acq_opt_mode == 'global' and self.acq_opt_method == 'turbo':
            self.update_groups()
            turbo1 = Turbo1_Component(
                f=lambda x, cur_group=None: -self.func(x, cur_group=cur_group),  # Handle to objective function
                lb=self.lb.cpu().numpy(),  # Numpy array specifying lower bounds
                ub=self.ub.cpu().numpy(),  # Numpy array specifying upper bounds
                n_init=1,  # unused parameter
                max_evals=20,  # Maximum number of evaluations     20 or 50 for more evaluations
                batch_size=1,  # How large batch size TuRBO uses
                verbose=False,  # Print information from each batch
                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                min_cuda=1,  # Run on the CPU for small datasets
                device=self.device,  # "cpu" or "cuda"
                dtype="float32",  # float64 or float32
            )
            Y_init = -np_train_y
            X_sample, Y_sample = turbo1.optimize(self.groups, np_train_x, Y_init, n=1, precision_record=self.precision_record)
            Y_sample = [-y for y in Y_sample]
            for new_x, value in zip(X_sample, Y_sample):
                new_samples.append((new_x, value))
                self.sample_counter += 1
                self.ema(value)
                if value > self.curt_best_value:
                    self.curt_best_value = value
                    self.curt_best_sample = new_x
                    self.best_value_trace.append((self.sample_counter, self.curt_best_value))
                self.value_trace.append((self.sample_counter, self.curt_best_value))
                self.samples.append((new_x, value))

        elif self.acq_opt_mode == 'global' and self.acq_opt_method == 'dscaled':
            self.update_groups()

            lb_full = self.lb.clone().to(self.device)
            ub_full = self.ub.clone().to(self.device)

            def build_additive_gp(
                    train_x,
                    train_y,
                    groups,
                    gp_params=None,
                    device=None,
                    dtype=self.tdtype,
                    use_ard=True,
            ):

                from gpytorch.likelihoods import GaussianLikelihood
                from gpytorch.kernels import MaternKernel, ScaleKernel, AdditiveKernel
                from gpytorch.priors import LogNormalPrior, GammaPrior
                from gpytorch.constraints import GreaterThan
                if gp_params is None:
                    gp_params = {
                        "const": {"loc": 0.0},
                        "ls": {"loc": 1.4, "scale": 1.73205},
                        "noise": {"loc": -4.0, "scale": 1.0},
                    }
                noise_params = gp_params.get('noise', {})
                gp_constraints = {}
                noise_constraint = gp_constraints.get('noise', 1e-4)

                kernels = []
                for g in groups:
                    ard_dims = len(g) if use_ard else None
                    ls_params = gp_params.get('ls', {})
                    ops_params = gp_params.get('ops', {})
                    noise_params = gp_params.get('noise', {})
                    ls_params['loc'] = ls_params['loc'] + math.log(self.dims) * 0.5
                    # toyed with scaling the scale parameter as well
                    ls_params['scale'] = (ls_params['scale'] ** 2 + math.log(self.dims) * 0) ** 0.5
                    ls_constraint = gp_constraints.get('ls', 1e-4)
                    scale_constraint = gp_constraints.get('scale', 1e-4)
                    noise_constraint = gp_constraints.get('noise', 1e-4)
                    """
                    base_kernel = MaternKernel(
                        nu=2.5,
                        active_dims=g,
                        ard_num_dims=ard_dims,
                        lengthscale_prior=LogNormalPrior(**ls_params),
                        lengthscale_constraint=GreaterThan(ls_constraint),
                    )
                    kernels.append(
                        ScaleKernel(
                            base_kernel,
                            outputscale_prior=GammaPrior(2.0, 0.15),
                        )
                    )
                    """
                    outputscale_prior = GammaPrior(
                        concentration=torch.tensor(2.0, device=self.device, dtype=self.tdtype),
                        rate=torch.tensor(0.15, device=self.device, dtype=self.tdtype)
                    )
                    lengthscale_prior = LogNormalPrior(
                        loc=torch.tensor(ls_params['loc'], device=self.device, dtype=self.tdtype),
                        scale=torch.tensor(ls_params['scale'], device=self.device, dtype=self.tdtype)
                    )

                    # 构建 kernel 和 likelihood
                    base_kernel = MaternKernel(
                        nu=2.5,
                        active_dims=g,
                        ard_num_dims=ard_dims,
                        lengthscale_prior=lengthscale_prior,
                        lengthscale_constraint=GreaterThan(ls_constraint),
                    )
                    kernels.append(
                        ScaleKernel(
                            base_kernel,
                            outputscale_prior=outputscale_prior,
                        )
                    )
                noise_prior = LogNormalPrior(
                    loc=torch.tensor(noise_params['loc'], device=self.device, dtype=self.tdtype),
                    scale=torch.tensor(noise_params['scale'], device=self.device, dtype=self.tdtype)
                )
                likelihood = GaussianLikelihood(
                    noise_prior=noise_prior,
                    noise_constraint=GreaterThan(noise_constraint),
                )
                likelihood = likelihood.to(self.device).to(dtype=self.tdtype)
                add_kernel = AdditiveKernel(*kernels).to(self.device).to(dtype=self.tdtype)
                gp_model = SingleTaskGP(X_train, Y_train.unsqueeze(-1), covar_module=add_kernel,likelihood=likelihood).to(self.device).to(
                    dtype=self.tdtype)
                return gp_model

            X_train = torch.from_numpy(np.array(train_x)).to(dtype=self.tdtype).to(self.device)
            Y_train = torch.from_numpy(np.asarray(train_y)).to(device=self.device, dtype=self.tdtype).view(-1)
            model = build_additive_gp(
                X_train,
                Y_train,
                self.groups,
                gp_params=gp_params,
                device=self.device
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll, options={"maxiter": 50},disable_priors=True)
            acqf = ExpectedImprovement(
                model=model,
                best_f=Y_train.max().item(),
            )
            #proposed_X, _ = optimize_acquisition(acqf, self.dims, self.lb, self.ub, n_points=1, n_candidates=2048, device=
            proposed_X, _ = botorch_optimize_acqf(
                acq_function=acqf,
                bounds=torch.stack([lb_full, ub_full]),
                q=self.sample_batch_size,
                num_restarts=5,
                raw_samples=128,
            )
            for x in proposed_X:
                x = x.detach().cpu().numpy()
                if self.precision_record:
                    y = self.func(x, cur_group = self.groups)
                else:
                    y = self.func(x)

                self.ema(y)
                self.samples.append((x, y))
                new_samples.append((x, y))
                self.sample_counter += 1
                if y > self.curt_best_value:
                    self.curt_best_value = y
                    self.curt_best_sample = x
                    self.best_value_trace.append((self.sample_counter, self.curt_best_value))
                self.value_trace.append((self.sample_counter, self.curt_best_value))
            del X_train, Y_train, model, mll, acqf
            torch.cuda.empty_cache()

        else:
            self.fit_gp()
            if self.acqf_name.lower() == "ei":
                self.acqf = ExpectedImprovement(self.gp_model, best_f=self.curt_best_value)
            elif self.acqf_name.lower() == "ucb":
                self.acqf = UpperConfidenceBound(self.gp_model, beta=0.2)
            else:
                raise ValueError(f"Unknown acquisition function: {self.acqf_name}")
            #acqf = ExpectedImprovement(self.gp_model, best_f=self.curt_best_value)
            lb_full = self.lb.clone().to(self.device)
            ub_full = self.ub.clone().to(self.device)
            if self.acq_opt_mode == 'global':
                if self.acq_opt_method == 'sobol':
                    X_new, _ = self.optimize_acqf_sobol(
                        self.acqf, self.gp_model, self.curt_best_value,
                        lb_full, ub_full,
                        n_points=self.sample_batch_size,
                        n_candidates = 1024,
                    )
                    X_new = X_new.detach().cpu().numpy()
                elif self.acq_opt_method == 'botorch':
                    X_new, _ = botorch_optimize_acqf(
                        acq_function=self.acqf,
                        bounds=torch.stack([lb_full, ub_full]),
                        q=self.sample_batch_size,
                        num_restarts=5,
                        raw_samples=100,
                    )
                    X_new = X_new.detach().cpu().numpy()

            elif self.acq_opt_mode == 'group':
                #X_train, Y_train = zip(*self.samples)
                #X_train = torch.from_numpy(np.array(X_train)).to(dtype=self.tdtype).to(self.device)
                #x_ref = self.curt_best_sample.copy() if self.curt_best_sample is not None \
                #        else np.mean(np.array(X_train.cpu()), axis=0)
                x_ref, _ = self.optimize_acqf_sobol(self.acqf, self.gp_model, self.curt_best_value, self.lb, self.ub, n_points=1, n_candidates=2048)
                x_ref = x_ref[0].cpu().numpy()
                X_new = []
                num = 1024
                for g in self.groups:
                    lb_opt, ub_opt = lb_full.clone(), ub_full.clone()
                    for i in range(self.dims):
                        if i not in g:
                            lb_opt[i] = ub_opt[i] = float(x_ref[i])
                    if self.acq_opt_method == 'sobol':
                        X_sub, _ = self.optimize_acqf_sobol(
                            self.acqf, self.gp_model, self.curt_best_value,
                            lb_opt, ub_opt, n_points=self.sample_batch_size, n_candidates=num
                        )
                        X_sub = X_sub.detach().cpu().numpy()
                    elif self.acq_opt_method == 'botorch':
                        X_sub, _ = botorch_optimize_acqf(
                            acq_function=self.acqf,
                            bounds=torch.stack([lb_opt, ub_opt]),
                            q=self.sample_batch_size,
                            num_restarts=5,
                            raw_samples=100,
                        )
                        X_sub = X_sub.detach().cpu().numpy()

                    if self.sample_batch_size == 1:
                        x_ref = X_sub[0].copy()
                    else:
                        X_t = torch.tensor(X_sub, dtype=self.tdtype).to(self.device)
                        with torch.no_grad():
                            ei_vals = self.acqf(X_t).squeeze()
                        best_idx = int(torch.argmax(ei_vals))
                        x_ref = X_sub[best_idx].copy()
                X_new = X_sub

            for x in X_new:
                if self.precision_record:
                    y = self.func(x, cur_group = self.groups)
                else:
                    y = self.func(x)
                self.ema(y)
                self.samples.append((x, y))
                self.sample_counter += 1

                if y > self.curt_best_value:
                    self.curt_best_value = y
                    self.curt_best_sample = x
                    self.best_value_trace.append((self.sample_counter, self.curt_best_value))
                self.value_trace.append((self.sample_counter, self.curt_best_value))
                new_samples.append((x, y))

        return new_samples

    def search(self, max_samples, verbose=True):
        idx = 0
        while True:
            if verbose:
                print('\n' + '='*10)
                print('iteration: {}'.format(idx))
                print('='*10)
            new_samples = self.collect_samples()
            if self.metric_classes == "AverageImprovementInteraction":
                for x, y in new_samples:
                    for group in self.groups:
                        for i in range(len(group)):
                            for j in range(i + 1, len(group)):
                                a, b = group[i], group[j]
                                improvement = y
                                #self.interaction_sum[a, b] = 0.65 * improvement + 0.25 * self.interaction_sum[a, b]
                                #self.interaction_sum[b, a] = 0.65 * improvement + 0.25 * self.interaction_sum[b, a]
                                self.interaction_sum[a, b] += improvement
                                self.interaction_sum[b, a] += improvement
                                self.interaction_count[a, b] += 1
                                self.interaction_count[b, a] += 1
            if verbose:
                print('New samples collected: {}'.format(len(new_samples)))
                print('Total samples so far: {}'.format(len(self.samples)))
                print('Current best value: {:.6f}'.format(self.curt_best_value))
            idx += 1
            self.t = idx
            if self.sample_counter >= max_samples:
                if verbose:
                    print('Reached max_samples: {}, stopping search.'.format(max_samples))
                break
