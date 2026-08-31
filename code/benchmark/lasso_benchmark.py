import numpy as np
import LassoBench

class HpoLassoBench:
    """
    Simple wrapper for LassoBench to use as a benchmark function.
    Provides continuous input space [0,1]^d.
    """
    def __init__(self, pick_data='DNA'):
        """
        pick_data: str, dataset name, e.g., 'leukemia', 'rcv1'
        """
        # Initialize LassoBench RealBenchmark
        self.benchmark = LassoBench.RealBenchmark(pick_data=pick_data, mf_opt="discrete_fidelity")

        # Number of features = optimization dimensions
        self.dims = self.benchmark.n_features

        self.lb = -np.ones(self.dims)
        self.ub = np.ones(self.dims)

    def __call__(self, x):
        """
        Evaluate the benchmark.
        x: np.array of shape (dims,), values in [0,1]
        """
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        return -self.benchmark.fidelity_evaluate(x, index_fidelity=0)


