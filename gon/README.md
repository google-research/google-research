Example code for running Global Optimization Networks (GON) algorithm.

Paper: https://arxiv.org/abs/2202.01277 (ICML'22)

Abstract: We consider the problem of estimating a good maximizer of a black-box
function given noisy examples. We propose to fit a new type of function called a
global optimization network (GON), defined as any composition of an invertible
function and a unimodal function, whose unique global maximizer can be inferred
in O(D) time, and used as the estimate. As an example way to construct GON
functions, and interesting in its own right, we give new results for specifying
multi-dimensional unimodal functions using lattice models with linear inequality
constraints. We extend to conditional GONs that find a global maximizer
conditioned on specified inputs of other dimensions. Experiments show the GON
maximizers are statistically significantly better predictions than those
produced by convex fits, GPR, or DNNs, and form more reasonable predictions for
real-world problems.

Example run:

`python -m gon.simulation`
