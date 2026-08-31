import argparse
import numpy as np
from src.evaluator import Evaluator

class edaBench:
    def __init__(self, placer, benchmark, eval_gp_hpwl=False):
        args = argparse.Namespace(
            placer=placer,
            benchmark=benchmark,
            eval_gp_hpwl=False,
        )
        evaluator = Evaluator(args)
        self.dim = evaluator.n_dim
        xl = evaluator.xl.tolist()
        xu = evaluator.xu.tolist()
        assert len(xl) == len(xu) == self.dim
        self.lb = np.asarray(xl)
        self.ub = np.asarray(xu)
        self.evaluator = Evaluator(args)


    def __call__(self, x):
        assert isinstance(x, np.ndarray)
        assert x.shape == (self.dims,)
        assert np.all(x >= self.lb) and np.all(x <= self.ub)

        eda_values = self.evaluator.evaluate(x)
        return -eda_values
