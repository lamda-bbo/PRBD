# Scaling Bayesian Optimization to High-Dimension via Pairwise-Relation-Based Decomposition


## Requirements

- Ubuntu == 22.14
- Python == 3.8.13
- PyTorch == 2.4.0
- BoTorch == 0.8.5
- [NAS-Bench-101](https://github.com/google-research/nasbench)
- NAS-Bench-1Shot1 in [HPO-Bench](https://github.com/automl/HPOBench)
- NAS-Bench-201, TransNAS-Bench-101, NAS-Bench-ASR in [NASLib](https://github.com/automl/NASLib)

## Usage

Run ```bash scripts/start.sh``` to evaluate PRBD and other baselines. 



## Reference

BO, MCTS-VS, Turbo:

[GitHub - lamda-bbo/MCTS-VS: Official implementation of NeurIPS'22 paper "Monte Carlo Tree Search based Variable Selection for High-Dimensional Bayesian Optimization"](https://github.com/lamda-bbo/MCTS-VS)

RDUCB, Tree:

[HEBO/RDUCB at master · huawei-noah/HEBO · GitHub](https://github.com/huawei-noah/HEBO/tree/master/RDUCB)

BOIDS:

[GitHub - LamNgo1/boids: [AAAI' 25\] BOIDS: High-dimensional Bayesian Optimization via Incumbent-guided Direction Lines and Subspace Embeddings](https://github.com/LamNgo1/boids)

D_scale:

[GitHub - hvarfner/vanilla_bo_in_highdim](https://github.com/hvarfner/vanilla_bo_in_highdim/tree/main)
