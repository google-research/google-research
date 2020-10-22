# Approximate Heavily-Constrained Learning with Lagrange Multiplier Models

Example code for running experiments in the paper:
H. Narasimhan, A. Cotter, Y. Zhou, S. Wang, W. Guo, "Approximate
Heavily-Constrained Learning with Lagrange Multiplier Models", NeurIPS 2020.

Run the following to install dependencies:
```shell
virtualenv -p python3 .
source ./bin/activate

pip3 install -r requirements.txt
```

Run the following for the intersectional fairness experiments.
```shell
python3 -m intersectional_fairness -- --num_layers=1 --num_nodes=50
```

Run the following for the cross-group ranking fairness experiments with 
per-query constraints.
```shell
python3 -m ranking_fairness
```

Run the following for the fairness experiments with noisy protected groups.
```shell
python3 -m robust_fairness
```

