import torch
import botorch
import numpy as np
import argparse
import random
from benchmark import get_problem
from IndexAdditiveBO.IndexAdditiveBO import IndexAdditiveBO
from utils import save_args
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--func', default='add_f120_20', type=str)
parser.add_argument('--max_samples', default=160, type=int)
parser.add_argument('--max_group_size', default=5, type=int)
parser.add_argument('--sample_init_size', default=10, type=int)
parser.add_argument('--sample_batch_size', default=3, type=int)
parser.add_argument('--kernel_type', default='Matern', type=str)
parser.add_argument('--acq_opt_mode', default='group', type=str)
parser.add_argument('--acq_opt_method', default='sobol', type=str)
parser.add_argument('--group_update_interval', default=1, type=int)
parser.add_argument('--metric_type', default='single', type=str)
parser.add_argument('--metric_classes',nargs='+',default=['AverageImprovementInteraction'], type=str)
parser.add_argument('--grouping_strategy', default='Kmeans', type=str)
parser.add_argument('--weights',nargs='+',default=None,type=int)
parser.add_argument('--weight_strategy', default='Learn', type=str)
parser.add_argument('--root_dir', default='synthetic_logs', type=str)
parser.add_argument('--dir_name', default=None, type=str)
parser.add_argument('--postfix', default=None, type=str)
parser.add_argument('--precision_record', default="no", type=str)
parser.add_argument('--device', default="cuda:1", type=str)
parser.add_argument('--seed', default=2018, type=int)
args = parser.parse_args()

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

tmp = args.metric_classes[0] if args.metric_type == "single" else "multi"
if args.grouping_strategy.lower() == "true":
    algo_name = f"additive_bo_{args.grouping_strategy}_{args.acq_opt_mode}_{args.acq_opt_method}"
elif args.grouping_strategy.lower() == "random":
    algo_name = f"additive_bo_{args.grouping_strategy}_{args.max_group_size}_{args.acq_opt_mode}_{args.acq_opt_method}"
elif args.grouping_strategy.lower() == "tree":
    algo_name = f"additive_bo_{args.grouping_strategy}_{args.acq_opt_mode}_{args.acq_opt_method}"
elif args.grouping_strategy.lower() == "rtree":
    algo_name = f"additive_bo_{args.grouping_strategy}_{args.acq_opt_mode}_{args.acq_opt_method}"
else:
    tmp = args.metric_classes[0] if args.metric_type == "single" else "multi"
    algo_name = f"additive_bo_{args.grouping_strategy}_{args.max_group_size}_{tmp}_{args.acq_opt_mode}_{args.acq_opt_method}"

if args.postfix is not None:
    algo_name += ('_' + args.postfix)
save_config = {
    'save_interval': 40,
    'root_dir': 'logs/' + args.root_dir,
    'algo': algo_name,
    'func': args.func if args.dir_name is None else args.dir_name,
    'seed': args.seed
}
f = get_problem(args.func, save_config, args.seed)

save_args(
    'config/' + args.root_dir,
    algo_name,
    args.func if args.dir_name is None else args.dir_name,
    args.seed,
    args
)


agent = IndexAdditiveBO(
    func=f,
    dims=f.dims,
    lb=f.lb,
    ub=f.ub,
    true_groups=f.true_groups,
    max_group_size=args.max_group_size,
    sample_init_size=args.sample_init_size,
    sample_batch_size=args.sample_batch_size,
    kernel_type=args.kernel_type,
    acq_opt_mode=args.acq_opt_mode,
    acq_opt_method=args.acq_opt_method,
    group_update_interval=args.group_update_interval,
    metric_type=args.metric_type,
    metric_classes=args.metric_classes[0] if args.metric_type == "single" else args.metric_classes,
    grouping_strategy=args.grouping_strategy,
    weights=args.weights,
    weight_strategy=args.weight_strategy,
    precision_record = True if args.precision_record.lower() == "yes" else False,
    device = args.device
)

agent.search(max_samples=args.max_samples, verbose=False)

print('best f(x):', agent.value_trace[-1][1])