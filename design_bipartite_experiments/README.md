# Design and Analysis of Bipartite Experiments under a Linear Exposure-Response Model

C++20 implementation of the balanced correlation clustering heuristic from
Section 7.3 of [Design and Analysis of Bipartite Experiments under a Linear
Exposure-Response Model](https://arxiv.org/abs/2103.06392) (Harshaw et al.,
2021).

The main interface is `exposure_design.h`. A simple driver script,
`exposure_design_main.cc`, is also provided. It reads the matrix W from standard
input and parameters from the environment. It writes the resulting clustering to
standard output, preceded by the value of the objective. Sample usage in
`run.sh`.

Sample output:

```
10109.1
div1 div3
div2
div4 div5
div6
```
