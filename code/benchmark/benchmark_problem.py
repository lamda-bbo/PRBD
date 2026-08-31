import numpy as np
import pandas as pd
import re
import time
from benchmark.synthetic_function import Ackley, Rastrigin,Branin, Hartmann, Levy, Rosenbrock, HartmannExtend, f1 , f2, f3, f4, f0, f01,f02,f03,f04,f5,Rastrigin,f_ov1,f_ov2,f_ov3,f_ov4,f_ov5,f_ov6
from benchmark.tracker import Tracker, save_results
from benchmark.rover_function import Rover



class FunctionBenchmark:
    def __init__(self, func, dims, valid_idx, save_config):
        assert func.dims == len(valid_idx)
        self.func = func
        self.dims = dims
        self.valid_idx = valid_idx
        self.lb = func.lb[0] * np.ones(dims)
        self.ub = func.ub[0] * np.ones(dims)
        #self.true_groups = func.true_groups
        self.true_groups = getattr(func, "true_groups", None)

        self.save_config = save_config
        self.tracker = Tracker(
            self.save_config['save_interval'],
            self.save_config
        )
        
    def __call__(self, x, cur_group = None):
        assert len(x) == self.dims
        result = self.func(x[self.valid_idx])
        if cur_group != None:
            if cur_group == 1:
                same_1 = -1
                same_0 = -1
                score = -1
            else:
                def groups_to_adj(groups, dims):
                    A = np.zeros((dims, dims), dtype=int)
                    for g in groups:
                        for i in g:
                            for j in g:
                                A[i, j] = 1
                    return A
                A_true = groups_to_adj(self.true_groups, self.dims)
                A_pred = groups_to_adj(cur_group, self.dims)
                same_1 = np.sum((A_true == 1) & (A_pred == 1))
                same_0 = np.sum((A_true == 0) & (A_pred == 0))
                score = (same_1 + same_0) / (self.dims * self.dims)
            self.tracker.track(result, same1=same_1,same0=same_0,score=score)
        else:
            self.tracker.track(result)
        return result
    
    
class RLBenchmark:
    def __init__(self, func, dims, valid_idx, save_config):
        assert func.dims == len(valid_idx)
        self.func = func
        self.dims = dims
        self.valid_idx = valid_idx
        self.lb = func.lb[0] * np.ones(dims)
        self.ub = func.ub[0] * np.ones(dims)
        self.save_config = save_config

        self.true_groups = getattr(func, "true_groups", None)
        self.counter = 0
        self.n_samples = 0
        self.start_time = time.time()
        self.curt_best = float('-inf')
        self.best_value_trace = []
        
    def __call__(self, x, cur_group = None):
        assert len(x) == self.dims
        result, n_samples = self.func(x[self.valid_idx])
        self.track(result, n_samples)
        return result
    
    def track(self, result, n_samples):
        self.counter += 1
        self.n_samples += n_samples
        if result > self.curt_best:
            self.curt_best = result
        self.best_value_trace.append((
            self.counter,
            self.curt_best,
            time.time() - self.start_time,
            self.n_samples
        ))
        
        if self.counter % 50 == 0:
            df_data = pd.DataFrame(self.best_value_trace, columns=['x', 'y', 't', 'n_samples'])
            save_results(
                self.save_config['root_dir'],
                self.save_config['algo'],
                self.save_config['func'],
                self.save_config['seed'],
                df_data,
            )
    

hartmann6 = Hartmann(6, True)
levy10 = Levy(10, True)
f010 = f0(10,False)
f020 = f0(20, False)
#f120 = f1(20, False)
#f216 = f2(16, False)
#f320 = f3(20, False)
#f416 = f4(16, False)
#f010 = f0(10, False)
f0140 = f01(40, False)
f0160 = f01(60, False)
f0180 = f01(80, False)
f0240 = f02(40, False)
f0260 = f02(60, False)
f0280 = f02(80, False)
Ackley10 = Ackley(10,True)
Ackley20 = Ackley(20,True)
Ackley40 = Ackley(40,True)
Ackley60 = Ackley(60,True)
Ackley80 = Ackley(80,True)
f0310 = f03(10,False)
f0320 = f03(20,False)
f0340 = f03(40,False)
f0360 = f03(60,False)
f0380 = f03(80,False)
f0410 = f04(10,False)
f0420 = f04(20,False)
f0440 = f04(40,False)
f0460 = f04(60,False)
f0480 = f04(80,False)
Rastrigin10 = Rastrigin(10,True)
Rastrigin20 = Rastrigin(20,True)
Rastrigin40 = Rastrigin(40,True)
Rastrigin60 = Rastrigin(60,True)
Rastrigin80 = Rastrigin(80,True)

f1105 = f1(10,5,False)
f1205 = f1(20,5,False)
f14010 = f1(40,10,False)
f16010 = f1(60,10,False)
f18010 = f1(80,10,False)

f3105 = f3(10,5,False)
f3205 = f3(20,5,False)
f34010 = f3(40,10,False)
f36010 = f3(60,10,False)
f38010 = f3(80,10,False)

f5105 = f5(10,5,False)
f5205 = f5(20,5,False)
f54010 = f5(40,10,False)
f56010 = f5(60,10,False)
f58010 = f5(80,10,False)

fov1 =f_ov1(40,False)
fov2 =f_ov2(40,False)
fov3 =f_ov3(40,False)
fov4 =f_ov4(60,False)
fov5 =f_ov5(60,False)
fov6 =f_ov6(60,False)



def get_problem(func_name, save_config, seed=2021):
    """
    save_config: {'save_interval': int, 'root_dir': str, 'algo': str, 'func': str, 'seed': int}
    """
    if func_name in ['nasbench', 'rover', 'HalfCheetah', 'Walker2d', 'Hopper']:
        if func_name in ['HalfCheetah', 'Walker2d', 'Hopper']:
            from benchmark.rl_benchmark import RLEnv
            
        if func_name == 'nasbench':
            from benchmark.nas_benchmark import NasBench
            return FunctionBenchmark(NasBench(seed=seed), 36, list(range(36)), save_config)
        elif func_name == 'rover':
            return FunctionBenchmark(Rover(), 60, list(range(60)), save_config)
        elif func_name == 'HalfCheetah' or func_name == 'Walker2d':
            return RLBenchmark(RLEnv(func_name+'-v5', seed), 102, list(range(102)), save_config)
        elif func_name == 'Hopper':
            return RLBenchmark(RLEnv('Hopper-v5', seed), 33, list(range(33)), save_config)
        else:
            assert 0
    elif func_name in ['Lasso']:
        from benchmark.lasso_benchmark import HpoLassoBench
        return FunctionBenchmark(HpoLassoBench(), 180, list(range(180)), save_config)

    elif func_name in ['SVM']:
        from benchmark.svm_benchmark import HpoSVMBench
        return FunctionBenchmark(HpoSVMBench(), 388, list(range(388)), save_config)
    elif func_name != 'nasbench' and func_name.startswith('nasbench'):
        if func_name == 'nasbench1shot1':
            from benchmark.hpo_benchmark import HpoNasBench
            dims = 33
            return FunctionBenchmark(HpoNasBench(name=func_name, seed=seed), dims, list(range(dims)), save_config)
        elif func_name == 'nasbench201':
            from benchmark.naslib_benchmark import NASLibBench
            dims = 30
            return FunctionBenchmark(NASLibBench(name=func_name, seed=seed), dims, list(range(dims)), save_config)
        elif func_name == 'nasbenchtrans':
            from benchmark.naslib_benchmark import NASLibBench
            dims = 24
            return FunctionBenchmark(NASLibBench(name='transbench101_micro', seed=seed), dims, list(range(dims)), save_config)
        elif func_name == 'nasbenchasr':
            from benchmark.naslib_benchmark import NASLibBench
            dims = 30
            return FunctionBenchmark(NASLibBench(name='asr', seed=seed), dims, list(range(dims)), save_config)
        else:
            assert 0
    else:
        split_result = func_name.split('_')

        if func_name.startswith('add'):
            # add_f120_20 或 add_f1
            func = split_result[1]  # f1
            dims = int(split_result[2]) if len(split_result) == 3 else None
            return FunctionBenchmark(eval(func), dims, list(range(dims)), save_config)

        elif len(split_result) == 1:
            func = split_result[0]
            dims = None
        elif len(split_result) == 2:
            func, dims = split_result
            dims = int(dims)
        else:
            assert 0

        valid_dims = int(re.findall(r'\d+', func)[0])
        dims = valid_dims if dims is None else dims
        if func_name.startswith('hartmann') and split_result[0] != 'hartmann6':
            d = split_result[0].strip('hartmann')
            valid_dims = int(d)
            dims = int(func_name.split('_')[-1])
            return FunctionBenchmark(HartmannExtend(valid_dims, True), dims, list(range(valid_dims)), save_config)
        return FunctionBenchmark(eval(func), dims, list(range(valid_dims)), save_config)


if __name__ == '__main__':
    x = np.random.randn(50)
    save_config = {
        'save_interval': 3,
        'root_dir': 'logs',
        'algo': 'bo',
        'func': 'lecy10_50',
        'seed': 42
    }
    func = get_problem('levy10_50', save_config)
    
    for _ in range(10):
        func(x)
