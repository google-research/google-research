# Differentiable Top-K operators

This project contains the implementation of differentiable top-K operators from the following papers:

- [1] [Differentiable Top-k Operator with Optimal Transport](https://arxiv.org/abs/2002.06504)
- [2] [A Framework For Differentiable Discovery Of Graph Algorithms](https://openreview.net/forum?id=5UvvKsBTDcR)



## Differentiable Top-K with Optimal Transport


Please refer to `soft_ot.py` in the current folder for the implementation using
optimal transport technique.

Also the `demo.ipynb` contains an end-to-end example for using this operator for
learning kNN.


## Top-K as black-box linear combinatorial optimization


As the top-K operator is equivalent to the following integer linear programming:

<img src="https://render.githubusercontent.com/render/math?math=\argmax_{x \in \{0,1\}}^M \sum_{j=1}^M c_jx_j, s.t. \sum_{j=1}^M x_j=K">

Then we can directly use the technique proposed in [Differentiation of Blackbox Combinatorial Solvers](https://arxiv.org/abs/1912.02175)
to solve the above optimization problem. This connection is explictly made in [2].


Please refer to `soft_ilp.py` in the curretn folder for the implementation using the above technique.


## Remark

Besides the technical differences, the `soft_ilp.py` will produce the discrete `0/1`
indicator of whether a certain choise is in top-K, while `soft_ot.py` would produce
fuzzy continouous probability outcome.
