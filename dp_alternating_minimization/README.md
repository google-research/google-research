# Differentially Private Alternating Minimization

This code accompanies the papers

- [Private Alternating Least Squares](https://proceedings.mlr.press/v139/chien21a.html)
- [Multi-Task Differential Privacy Under Distribution Skew](https://arxiv.org/abs/2302.07975)

### Algorithms
This library provides algorithms for training personalized models with
differentially private alternating minimization (DPAM).

[DP personalized models](https://proceedings.neurips.cc/paper/2021/hash/f8580959e35cb0934479bb007fb241c2-Abstract.html)
are of the form

$$f_i(x) = \langle u_i, v(x) \rangle$$

where $$u_i$$ is a user embedding (specific to each user i), and $$v(x)$$ is an
item encoder shared among all users. (Notice that each user i has a different
prediction $$f_i$$.) Applications include recommendation systems, where the
encoder $$v(x)$$ learns a (shared) representation of the items to recommend,
the user embedding $$u_i$$ encodes the user-specific preferences, and the dot
product $$\langle u_i, v(x) \rangle$$ captures the relevance of item $$x$$ to
user $i$. These models are also known as multi-encoder models and are important
in [federated learning](https://arxiv.org/abs/2102.03448).

DP alternating minimization alternates between:

1. Solving for the user embeddings $$u_i$$, using (exact) least squares
2. Training the shared encoder $$v(x)$$, using DP.
  - An important special case is matrix completion, where the encoder $$v(x)$$ is
linear. In this case, step 2) is solved using DP least squares, see
Algorithm 1 in [DPALS](https://proceedings.mlr.press/v139/chien21a.html).
  - For general encoders, step 2) is solved using DPSGD.

### Distribution Skew and Adaptive Budget Allocation
In practice, there is typically a long tail of items that have much fewer
training examples (in particular for recommendation problems). It has been known that private models are particularly
susceptible to this type of distribution skew.

The paper [Multi-Task Differential Privacy Under Distribution Skew](https://arxiv.org/abs/2302.07975)
proposes a method to properly handle the distribution skew, by adaptively
allocating the privacy budget of each user among the different items (see
Algorithms 1 and 2 in the paper). At a high level, by allocating more privacy
budget to tail items, one can improve the overall quality.

The library provides an implementation of this adaptive budget allocation.

### Colab notebook
The notebook `dpam.ipynb` provides examples for training (non-private and private)
models on the MovieLens benchmarks used in the two papers, along with tuned
hyper-parameters at different $$\epsilon$$ levels. [Link to open in colab](https://colab.research.google.com/github/google-research/google-research/blob/master/dp_alternating_minimization/movielens/dpam.ipynb)

