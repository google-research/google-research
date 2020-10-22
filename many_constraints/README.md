# Heavily-Constrained Learning with Lagrange Multiplier Models

Example code for running experiments in the paper:
H. Narasimhan, A. Cotter, Y. Zhou, S. Wang, W. Guo, "Approximate
Heavily-Constrained Learning with Lagrange Multiplier Models".

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
