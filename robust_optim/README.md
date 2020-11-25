# Robust Optimization Project

This repository contains the evaluation of the robustness of solutions found
using various optimization methods to adversarial perturbations.

Work in progress.

To train and evaluate a linear model using gradient descent and evaluate against
Linf attack run:

```
python -m robust_optim.train --config.dim 100 --config.num_train 10
--config.num_test 400 --config.optim.lr 0.1 --config.temperature 0.0001
--config.adv.lr 0.1 --config.adv.norm_type linf --config.adv.eps_tot 0.5
--config.adv.eps_iter 0.5 --config.adv.niters 100 --config.log_keys
'("risk/train/loss","risk/train/zero_one","risk/test/zero_one","risk/train/adv/linf","weight/norm/l1")'
--config.optim.name gd --config.optim.niters 1000 --config.model.arch linear
--config.model.regularizer none --config.log_interval 1
```
