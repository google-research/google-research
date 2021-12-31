
JAX code to accompany the NeurIPS paper ["Learning Generalized Gumbel-max Causal Mechanisms"](https://arxiv.org/abs/2111.06888), with authors Guy Lorberbom*, Daniel D. Johnson*, Chris J. Maddison, Daniel Tarlow, and Tamir Hazan.

This directory contains implementations of both Gadget 1 and Gadget 2 in JAX,
along with code to reproduce the experiments in Section 7.1 and the first two
columns of Table 1.

For the implementation used in the last two columns of Table 1 and for the experiments in Section 7.3, see https://github.com/GuyLor/gumbel_max_causal_gadgets_part2.

Contents:
- `tutorial.ipynb` gives an overview of the two gadgets
  and shows how they can be trained.
- `paper_experiments.ipynb` contains code to reproduce the
  results of the experiments.
- `coupling_util.py` contains various helper functions for constructing
  couplings, including the baselines we compare against.
- `gadget_1.py` contains an implementation of our first gadget.
- `gadget_2.py` contains an implementation of our second gadget.
- `experiment_util.py` contains helper functions and classes to support training
  our couplings. These were used to conduct our experiments.

---

If you use the code in this repository, please cite the following paper:
```
@inproceedings{lorberbom2021generalized,
author    = {Guy Lorberbom and
             Daniel D. Johnson and
             Chris J. Maddison and
             Daniel Tarlow and
             Tamir Hazan},
title     = {Learning Generalized Gumbel-max Causal Mechanisms},
booktitle = {Advances in Neural Information Processing Systems},
year      = {2021}
}
```