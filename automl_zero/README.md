# AutoML-Zero

This directory contains the open-sourced code for the paper:

*E. Real\*, C. Liang\*, D. R. So, and Q. V. Le. "AutoML-Zero: Evolving Machine Learning Algorithms From Scratch" (\*equal contribution)*

&nbsp;

## Quick 1-Minute Demo: Rediscovering Linear Regression from Scratch

As a quick demo (~1 minute in 1 CPU), we provide a script that runs a
local evolution experiment under simplified conditions:

* the goal is to discover an algorithm to solve linear tasks,
* using only the necessary ops for a linear regressor (i.e. a much smaller search space than most of the paper),
* starting with a population of completely random algorithms.

The demo will run short evolution experiments in succession on the "search
tasks" (see paper) and evaluate the best algorithm discovered by each
experiment on the "select tasks". Once an algorithm attains a
fitness (1 - RMS error) greater than 0.9999, it is selected for a final
evaluation on unseen data. To conclude, the demo prints the results of this
one-time final evaluation and shows the code for the corresponding algorithm.

Please install with:

```
TODO(crazydonkey): add command lines.
```

and run with:

```
TODO(crazydonkey): add command lines.
```

Repeated runs will use different random seeds. Note that the runtime may vary
widely due to the random initial conditions and hardware.

&nbsp;

## Reproducing Baselines

The following command can be used to reproduce the results in Supplementary
Section 9 ("Baselines") with the "Basic" method on 1 process (1 CPU):

*[To be continued, ETA: March, 2020]*

If you want to use more than 1 process, you will need to code a way to
parallelize the computation based on your particular distributed-computing
platform. A platform-agnostic description of what we did is given in our paper.

We left out of this directory upgrades for the "Full" method that are
pre-existing (hurdles) but included those introduced in this paper (e.g. FEC
for ML algorithms).

&nbsp;

<sup><sub>
Search keywords: machine learning, neural networks, evolution,
evolutionary algorithms, regularized evolution, program synthesis,
architecture search, NAS, neural architecture search,
neuro-architecture search, AutoML, AutoML-Zero, algorithm search,
meta-learning, genetic algorithms, genetic programming, neuroevolution,
neuro-evolution.
</sub></sup>
