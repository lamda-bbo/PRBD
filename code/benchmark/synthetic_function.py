from abc import ABCMeta
import numpy as np


class SyntheticFunction(metaclass=ABCMeta):
    def __init__(self, dims, negate, lb, ub, opt_val, opt_point, true_groups = None):
        self.dims = dims
        self.negate = negate
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.opt_val = opt_val
        self.opt_point = np.array(opt_point)
        self.true_groups = true_groups
        
    def __call__(self, x):
        pass

class f0(SyntheticFunction):
    def __init__(self, dims=20, negate=False):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            [ list(range(dims)) ]
        )

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = 0.0
        result = np.prod(x)/40.0
        if self.negate:
            result = - result
        return result

class f01(SyntheticFunction):
    def __init__(self, dims=20, negate=False):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            [ list(range(dims)) ]
        )

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = 0.0
        half = len(x) // 2
        result = (np.sum(x[:half]) - np.sum(x[half:])) ** 2
        if self.negate:
            result = - result
        return result

class f02(SyntheticFunction):
    def __init__(self, dims=20, negate=False):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            [ list(range(dims)) ]
        )

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = 0.0
        half = len(x) // 2
        result = (np.sum(x[:half]) - np.sum(x[half:])) ** 3
        if self.negate:
            result = - result
        return result

class f03(SyntheticFunction):
    def __init__(self, dims=20, negate=False):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            [[i] for i in range(dims)]
        )

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = 0.0
        half = len(x) // 2
        result = (np.sum(x[:half]) - np.sum(x[half:]))
        if self.negate:
            result = - result
        return result

class f04(SyntheticFunction):
    def __init__(self, dims=20, negate=False):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            [[i] for i in range(dims)]
        )

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = 0.0
        half = len(x) // 2
        result = np.sum(x[:half]**2) - np.sum(x[half:]**2)
        if self.negate:
            result = - result
        return result


class Rastrigin(SyntheticFunction):
    def __init__(self, dims=20, negate=True):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -5.12 * np.ones(dims),
            5.12 * np.ones(dims),
            0,
            np.zeros(dims),
            [[i] for i in range(dims)]
        )

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = 10 * self.dims + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        if self.negate:
            result = -result
        return result




class f1(SyntheticFunction):
    def __init__(self, dims=20, per_group_num=5, negate=False):
        assert dims % 5 == 0
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            [list(range(i, i + per_group_num)) for i in range(0, dims, per_group_num)]
        )
        self.per_group_num = per_group_num

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = 0.0
        for i in range(0, self.dims, self.per_group_num):
            block = x[i:i+self.per_group_num]
            result += np.prod(block)
        if self.negate:
            result = - result
        return result

class f2(SyntheticFunction):
    def __init__(self, dims=16, negate=False):
        assert dims % 4 == 0
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            [list(range(i, i + 4)) for i in range(0, dims, 4)]
        )

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = 0.0
        for i in range(0, self.dims, 4):
            block = x[i:i+4]
            result += np.prod(block)
        if self.negate:
            result = - result
        return result

class f3(SyntheticFunction):
    def __init__(self, dims=20, per_group_num=5, negate=False):
        assert dims % 5 == 0
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -5 * np.ones(dims),
            5 * np.ones(dims),
            0,
            np.array([0] * dims),
            [list(range(i, i + per_group_num)) for i in range(0, dims, per_group_num)]
        )
        self.per_group_num = per_group_num

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = 0.0
        for i in range(0, self.dims, self.per_group_num):
            block = x[i:i+self.per_group_num]
            result += np.sum(block) ** 2
        if self.negate:
            result = - result
        return result

class f5(SyntheticFunction):
    def __init__(self, dims=20, per_group_num=5, negate=False):
        assert dims % 5 == 0
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -5 * np.ones(dims),
            5 * np.ones(dims),
            0,
            np.array([0] * dims),
            [list(range(i, i + per_group_num)) for i in range(0, dims, per_group_num)]
        )
        self.per_group_num = per_group_num

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = 0.0
        for i in range(0, self.dims, self.per_group_num):
            block = x[i:i+self.per_group_num]
            result += np.sum(block) ** 3
        if self.negate:
            result = - result
        return result

class f4(SyntheticFunction):
    def __init__(self, dims=16, negate=False):
        assert dims % 4 == 0
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            [list(range(i, i + 4)) for i in range(0, dims, 4)]
        )

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = 0.0
        for i in range(0, self.dims, 4):
            block = x[i:i+4]
            result += np.sum(block) ** 2
        if self.negate:
            result = - result
        return result


groups_40 = [
    [17, 3, 29, 8, 34, 11, 25, 6, 14, 31],
    [25, 6, 14, 31, 2, 19, 7, 33, 12, 28],
    [7, 33, 12, 28, 0, 22, 18, 9, 35, 4],
    [18, 9, 35, 4, 26, 15, 30, 21, 10, 36],
    [30, 21, 10, 36, 5, 24, 13, 32, 16, 27],
    [13, 32, 16, 27, 1, 23, 20, 37, 34, 11],
    [20, 37, 34, 11, 38, 39, 29, 8, 17, 3],
    [29, 8, 17, 3, 25, 6, 14, 31, 2, 19]
]

class f_ov1(SyntheticFunction):
    def __init__(self, dims=40, negate=False):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            groups_40
        )

    def __call__(self, x):
        assert len(x) == self.dims
        result = 0.0
        for g in groups_40:
            result += np.prod(x[g])
        if self.negate:
            result = -result
        return result

class f_ov2(SyntheticFunction):
    def __init__(self, dims=40, negate=False):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            groups_40
        )

    def __call__(self, x):
        assert len(x) == self.dims
        result = 0.0
        for g in groups_40:
            result += (np.sum(x[g]))**2
        if self.negate:
            result = -result
        return result

class f_ov3(SyntheticFunction):
    def __init__(self, dims=40, negate=False):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            groups_40
        )

    def __call__(self, x):
        assert len(x) == self.dims
        result = 0.0
        for g in groups_40:
            result += (np.sum(x[g]))**3
        if self.negate:
            result = -result
        return result

groups_60 = [
    [17, 3, 41, 9, 52, 1, 33, 28, 6, 14],
    [6, 14, 22, 48, 37, 55, 11, 2, 19, 7],
    [19, 7, 25, 31, 0, 44, 18, 8, 46, 29],
    [46, 29, 35, 4, 57, 12, 23, 10, 38, 21],
    [38, 21, 49, 15, 34, 27, 5, 40, 13, 53],
    [13, 53, 26, 58, 45, 16, 30, 24, 20, 36],
    [20, 36, 43, 51, 32, 47, 54, 42, 39, 56],
    [54, 42, 50, 59, 52, 1, 17, 3, 41, 9],
]

class f_ov4(SyntheticFunction):
    def __init__(self, dims=60, negate=False):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            groups_60
        )

    def __call__(self, x):
        assert len(x) == self.dims
        result = 0.0
        for g in groups_60:
            result += np.prod(x[g])
        if self.negate:
            result = -result
        return result
class f_ov5(SyntheticFunction):
    def __init__(self, dims=60, negate=False):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            groups_60
        )

    def __call__(self, x):
        assert len(x) == self.dims
        result = 0.0
        for g in groups_60:
            result += (np.sum(x[g]))**2
        if self.negate:
            result = -result
        return result
class f_ov6(SyntheticFunction):
    def __init__(self, dims=60, negate=False):
        SyntheticFunction.__init__(
            self,
            dims,
            negate,
            -2 * np.ones(dims),
            2 * np.ones(dims),
            0,
            np.array([0] * dims),
            groups_60
        )

    def __call__(self, x):
        assert len(x) == self.dims
        result = 0.0
        for g in groups_60:
            result += (np.sum(x[g]))**3
        if self.negate:
            result = -result
        return result


class Ackley(SyntheticFunction):
    def __init__(self, dims=20, negate=True):
        SyntheticFunction.__init__(
            self, 
            dims, 
            negate, 
            -10 * np.ones(dims),
            10 * np.ones(dims),
            0,
            np.array([0]*dims),
            [list(range(dims))]
        )
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        if self.negate:
            result = - result
        return result
    
    
class Branin(SyntheticFunction):
    def __init__(self, dims=2, negate=False):
        assert dims == 2
        SyntheticFunction.__init__(
            self,
            dims, 
            negate,
            np.array([-5, -5]),
            np.array([15, 15]),
            -0.397887,
            np.array([-np.pi, 12.275]), # [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]
        )
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        t1 = x[1] \
            - 5.1 / (4*np.pi**2) * x[0]**2 \
            + 5 / np.pi * x[0] - 6
        t2 = 10 * (1 - 1/(8*np.pi)) * np.cos(x[0])
        result = t1**2 + t2 + 10
        if self.negate:
            result = - result
        return result
    
    
class Hartmann(SyntheticFunction):
    def __init__(self, dims=6, negate=False):
        assert dims == 6
        SyntheticFunction.__init__(
            self, 
            dims, 
            negate, 
            0 * np.ones(dims),
            1 * np.ones(dims),
            -3.32237,
            np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        )
        
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ])
        self.P = np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ])
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        inner_sum = np.sum(self.A * (x.reshape(1, -1) - 0.0001 * self.P) ** 2, axis=-1)
        result = - np.sum(self.alpha * np.exp(-inner_sum), axis=-1)
        if self.negate:
            result = - result
        return result
    
    
class HartmannExtend(SyntheticFunction):
    def __init__(self, dims=30, negate=False):
        assert dims % 6 == 0
        SyntheticFunction.__init__(
            self, 
            dims, 
            negate, 
            0 * np.ones(dims),
            1 * np.ones(dims),
            -3.32237,
            np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        )
        
        self.func = Hartmann(6, negate)
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        result = 0
        for i in range(int(self.dims / 6)):
            result += self.func(x[i*6: (i+1)*6])
        return result
    
    
class Levy(SyntheticFunction):
    def __init__(self, dims=10, negate=False):
        SyntheticFunction.__init__(
            self, 
            dims, 
            negate, 
            -10 * np.ones(dims),
            10 * np.ones(dims),
            0,
            np.ones(dims)
        )
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        w = 1 + (x - 1.0) / 4.0
        result = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:self.dims - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dims - 1] + 1) ** 2)) + \
            (w[self.dims - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dims - 1])**2)
        if self.negate:
            result = - result
        return result
    
    
class Rosenbrock(SyntheticFunction):
    def __init__(self, dims=2, negate=False):
        SyntheticFunction.__init__(
            self, 
            dims, 
            negate, 
            -5 * np.ones(dims),
            10 * np.ones(dims),
            0,
            np.ones(dims)
        )
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = np.sum(100 * (x[1: ] - x[: -1]**2)**2 + (x[: -1] - 1)**2)
        if self.negate:
            result = - result
        return result
    
    
if __name__ == '__main__':
    # x = np.random.randn(6)
    # func = Hartmann(6, True)
    # print(func(x))
    # import torch
    # from botorch.test_functions import Hartmann, Branin, Levy
    # func = Hartmann(6, negate=True)
    # print(func(torch.tensor(x)))
    
    func = Branin(2, True)
    print(func(np.array([-np.pi, 12.275])))
    print(func(np.array([np.pi, 2.275])))
    print(func(np.array([9.42478, 2.475])))
    
    func = Ackley(10, True)
    print(func(np.zeros(10)))
    
    func = Rosenbrock(10, True)
    print(func(np.ones(10)))