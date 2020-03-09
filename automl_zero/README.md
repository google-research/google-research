# AutoML-Zero

Open source code for the paper: \"[**AutoML-Zero: Evolving Machine Learning Algorithms From Scratch**](https://github.com/google-research/google-research/tree/master/automl_zero)"

| [Introduction](#what-is-automl-zero) | [Quick Demo](#5-minute-demo-discovering-linear-regression-from-scratch)| [Reproducing Search Baselines](#reproducing-search-baselines) | [Citation](#citation) |
|-|-|-|-|

## What is AutoML-Zero?

AutoML-Zero aims at automatically discovering computer programs to solve machine learning tasks, starting from empty or random programs and using only basic math operations. The goal is to simultaneously search for all aspects of an ML algorithm (e.g., the model structure and the learning strategy), while employing *minimal human bias*. Despite the challenging search space for AutoML-Zero, *Evolutionary Search* showed promising results by discovering linear regression, 2-layer neural network with backpropagation, and even algorithms better than hand designed baselines of comparable complexity. An example sequence of discoveries on binary classification tasks is shown below.

![GIF for the experiment progress](progress.gif)

More importantly, the evolved algorithms can be *interpreted*. Below is an analysis of the best evolved algorithm, which "invents" techniques like bilinear interactions, weight averaging, normalized gradient and adding noise to inputs.

![GIF for the interpretation of the best evolved algorithm](best_algo.gif)

Note that the algorithms shown above are already simplified and reordered for better readability. Details about more experiments and analyses can be found in the [paper](https://github.com/google-research/google-research/tree/master/automl_zero).


&nbsp;

## 5-Minute Demo: Discovering Linear Regression From Scratch

As a miniature "AutoML-Zero" experiment, let's try to automatically discover programs to solve linear regression tasks.

To get started, first install `bazel` following instructions [here](https://docs.bazel.build/versions/master/install.html), then run the demo with:

```
git clone https://github.com/google-research/google-research.git
cd google-research/automl_zero
./run_demo.sh
```


This script runs evolutionary search on 10 linear tasks (*T<sub>search</sub>* in the paper). After each experiment, it evaluates the best algorithm discovered on 100 new linear tasks (*T<sub>select</sub>* in the paper). Once an algorithm attains a fitness (1 - RMS error) greater than 0.9999, it is selected for a final evaluation on 100 *unseen tasks*. To conclude, the demo prints the results of the final evaluation and shows the code for the automatically discovered algorithm.

To make this demo quick, we use a much smaller search space: only the math operations necessary to implement linear regression are allowed and the programs are constrained to a short, fixed length. This way, the demo will typically discover programs similar to linear regression by gradient descent in under 5 minutes using 1 CPU (Note that the runtime may vary due to the random seeds and hardware). We saw similar discoveries in the unconstrained search space, although at a higher compute costs.

You can compare the automatically discovered algorithm with the solution from a human ML researcher (one of the authors):

```
def Setup():
  s2 = 0.001  # Init learning rate.

def Predict():  # v0 = features
  s1 = dot(v0, v1)  # Apply weights

def Learn():  # v0 = features; s0 = label
  s3 = s0 - s1  # Compute error.
  s4 = s3 * s1  # Apply learning rate.
  v2 = v0 * s4  # Compute gradient.
  v1 = v1 + v2  # Update weights.
```

In this human designed program, the ```Setup``` function establishes a learning rate, the ```Predict``` function applies a set of weights to the inputs, and the ```Learn``` function corrects the weights in the opposite direction to the gradient. In other words, a linear regressor trained with gradient descent. You might have noticed that the evolved programs may order the instructions very differently and usually contain many redundant instructions, which can make it challenging to interpret. See more details about how we address these problems in the [paper](https://github.com/google-research/google-research/tree/master/automl_zero#automl-zero).

&nbsp;

## Reproducing Search Baselines

First install `bazel` following instructions [here](https://docs.bazel.build/versions/master/install.html),then run the following command to reproduce the results in Supplementary
Section 9 ("Baselines") with the "Basic" method on 1 process (1 CPU):

*[To be continued, ETA: March, 2020]*

If you want to use more than 1 process, you will need to create your own implementation to
parallelize the computation based on your particular distributed-computing
platform. A platform-agnostic description of what we did is given in our paper.

We left out of this directory upgrades for the "Full" method that are
pre-existing (hurdles) but included those introduced in this paper (e.g. FEC
for ML algorithms).

## Citation

If you use the code in your research, please cite:

`TODO`

&nbsp;

<sup><sub>
Search keywords: machine learning, neural networks, evolution,
evolutionary algorithms, regularized evolution, program synthesis,
architecture search, NAS, neural architecture search,
neuro-architecture search, AutoML, AutoML-Zero, algorithm search,
meta-learning, genetic algorithms, genetic programming, neuroevolution,
neuro-evolution.
</sub></sup>
