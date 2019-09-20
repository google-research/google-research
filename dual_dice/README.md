# DualDICE

Code for DualDICE as described in `DualDICE: Behavior-Agnostic Estimation of
Discounted Stationary Distribution Corrections' by Ofir Nachum, Yinlam Chow, Bo
Dai, Lihong Li.

Paper available at https://arxiv.org/abs/1906.04733

If you use this codebase for your research, please cite the paper:

```
@article{nachum2019dualdice,
  title={DualDICE: Behavior-Agnostic Estimation of Discounted Stationary
  Distribution Corrections},
  author={Nachum, Ofir and Chow, Yinlam and Dai, Bo and Li, Lihong},
  journal={NeurIPS},
  year={2019}
}
```

## Basic Commands

Run DualDICE on a tabular version of GridWalk:

```
python -m dual_dice.run --logtostderr --env_name=grid --tabular_obs --tabular_solver
```

Run DualDICE on a non-tabular version of GridWalk:

```
python -m dual_dice.run --logtostderr --env_name=grid --notabular_obs --notabular_solver \
    --nu_learning_rate=0.0001 --zeta_learning_rate=0.001
```

## Additional Information

When tuning DualDICE, some things to keep in mind:
*   Optimization of nu and zeta at the same time can be unstable. Try utilizing
    the --deterministic_env flag, which trains nu on its own (without
    application of Fenchel duality).
*   Lower gamma can make optimization easier.
*   Otherwise, nu and zeta learning rate have the biggest impact on performance.

When evaluating DualDICE, keep in mind that the print-out of the Target (oracle)
rewards is an estimate! In a perfect world this would be calculated with
num_trajectories=\infty and max_trajectory_length=\infty. In practice, to get a
better estimate, just set these to a large value (e.g., 1000).

This is a minimal version of the code. For example, continuous-action
environments are not supported.
