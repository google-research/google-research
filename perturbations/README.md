# Differentiable Optimizers with Perturbations in Tensorflow

## Overview

We propose in this work a universal method to transform any optimizer in a
differentiable approximation. We provide a TensorFlow implementation,
illustrated here on some examples.

## Perturbed argmax

We start from an original optimizer, an `argmax` function, computed on an
example input `theta`.

```python
import tensorflow as tf
import perturbations

def argmax(x, axis=-1):
  return tf.one_hot(tf.argmax(x, axis=axis), tf.shape(x)[axis])
```

This function returns a one-hot corresponding to the largest input entry.

```python
>>> argmax(tf.constant([-0.6, 1.9, -0.2, 1.1, -1.0], dtype=tf.float32))
[0., 1., 0., 0., 0.]
```

It is possible to modify the function by creating a perturbed optimizer, using
Gumbel noise.

```python
pert_argmax = perturbations.perturbed(argmax,
                                      num_samples=1000000,
                                      sigma=0.5,
                                      noise='gumbel',
                                      batched=False)
```

```python
>>> theta = tf.constant([-0.6, 1.9, -0.2, 1.1, -1.0], dtype=tf.float32)
>>> pert_argmax_fn(theta)
[0.005539, 0.814773, 0.012393, 0.164778, 0.002517]
```

In this particular case, it is equal to the usual softmax with exponential
weights.

```python
>>> tf.nn.softmax(theta_single/sigma)
[0.00549293, 0.8152234 , 0.01222475, 0.16459079, 0.00246813]
```

### Batched version

The original function can accept a batch dimension, and is applied to every
element of the batch.

```python
theta_batch = tf.constant([[-0.6, 1.9, -0.2, 1.1, -1.0],
                           [-0.6, 1.0, -0.2, 1.8, -1.0]], dtype=tf.float32)
```

```python
>>> argmax(theta_batch)
[[0., 1., 0., 0., 0.],
 [0., 0., 0., 1., 0.]]
```

Likewise, if the argument `batched` is set to `True` (its default value), the
perturbed optimizer can handle a batch of inputs.

```python
pert_argmax = perturbations.perturbed(argmax,
                                      num_samples=1000000,
                                      sigma=0.5,
                                      noise='gumbel',
                                      batched=True)
```

```python
>>> pert_argmax_fn(theta_batch)
[[0.005419, 0.815392, 0.012309, 0.164405, 0.002475],
 [0.006703, 0.163617, 0.014623, 0.812042, 0.003015]]
```

It can be compared to its deterministic version, the softmax.

```python
>>> tf.nn.softmax(theta_batch / sigma)
[[0.00549293, 0.8152234 , 0.01222475, 0.16459079, 0.00246813],
 [0.00667923, 0.16385847, 0.0148649 , 0.8115962 , 0.00300117]]
```

### Decorator version

It is also possible to use the perturbed function as a decorator.

```python
@perturbations.perturbed(num_samples=1000000, sigma=0.5, noise='gumbel', batched=True)
def argmax(x, axis=-1):
  return tf.one_hot(tf.argmax(x, axis=axis), tf.shape(x)[axis])
```

```python
>>> argmax(theta_batch)
[[0.005614, 0.815336, 0.012241, 0.164281, 0.002528],
 [0.006787, 0.164137, 0.01498 , 0.811193, 0.002903]]
```

### Gradient computation

The Perturbed optimizers are differentiable, and the gradients can be computed
with stochastic estimation automatically. In this case, it can be compared
directly to the gradient of softmax.

```python
with tf.GradientTape() as tape:
  theta = tf.Variable(theta_batch)
  pert_argmax = pert_argmax_fn(theta)
  square_norm = square_norm_fn(pert_argmax)
grad_pert, = tape.gradient(square_norm, [theta])
```

```python
>>> grad_pert
[[-0.01569094,  0.40335307, -0.03419076, -0.34780994, -0.0049111 ],
 [-0.01836201, -0.344668  , -0.04183569,  0.4055121 , -0.0065396 ]]
```

Compared to the same computations with a softmax.

```python
with tf.GradientTape() as tape:
  theta = tf.Variable(theta_batch)
  soft_max = tf.nn.softmax(theta/sigma)
  square_norm = square_norm_fn(pert_argmax)
grad_soft, = tape.gradient(square_norm, [theta])
```

```python
>>> grad_soft
[[-0.01508078,  0.4022587 , -0.03323372, -0.3471379 , -0.00680609],
 [-0.01814435, -0.34210643, -0.03989429,  0.40834203, -0.00819694]]
```

## Perturbed OR

The OR function over the signs of inputs, that is an example of optimizer,
offers a well-interpretable visualization.

```python
def hard_or(x):
  s = tf.cast((tf.sign(x) + 1)  / 2.0, dtype=tf.bool)
  result = tf.math.reduce_any(s, axis=-1)
  return tf.cast(result, dtype=tf.float32) * 2.0 - 1.0
```

In the following batch of two inputs, both instances are evaluated as `True`
(value `1`).

```python
theta = tf.constant([[-5., 0.2],
                     [-5., 0.1]])
```

```python
>>> hard_or(theta)
[1., 1.]
```

Computing a perturbed OR operator over 1000 samples shows the difference in
value for these two inputs.

```python
pert_or = perturbations.perturbed(hard_or,
                                  num_samples=1000,
                                  sigma=0.1,
                                  noise='gumbel',
                                  batched=True)
```

```python
>>> pert_or(theta)
[1.   , 0.868]
```

This can be vizualized more broadly, for values between -1 and 1, as well as the
evaluated values of the gradient.
<img src="https://q-berthet.github.io/pictures/soft-or.png" width=900>

## Perturbed shortest path

This framework can also be easily applied to more complex optimizers, such as a
blackbox shortest paths solver (here the function `shortest_path`). We consider
a small example on 9 nodes, illustrated here with the shortest path between 0
and 8 in bold, and edge costs labels.<br>

<img src="https://q-berthet.github.io/pictures/graphb.png" width=500>

We also consider a function of the perturbed solution: the weight of this
solution on the edgebetween nodes **6** and **8**.

A gradient of this function with respect to a vector of four edge costs
(top-rightmost, between nodes 4, 5, 6, and 8) is automatically computed. This
can be used to increase the weight on this edge of the solution by changing
these four costs. This is challenging to do with first-order methods using only
an original optimizer, as its gradient would be zero almost everywhere.

```python
final_edges_costs = tf.constant([0.4, 0.1, 0.1, 0.1], dtype=tf.float32)
weights = edge_costs_to_weights(final_edges_costs)

@perturbations.perturbed(num_samples=100000, sigma=0.05, batched=False)
def perturbed_shortest_path(weights):
 return shortest_path(weights, symmetric=False)
```

We obtain a perturbed solution to the shortest path problem on this graph, an
average of solutions under perturbations on the weights.

```python
>>> perturbed_shortest_path(weights)
[[0.    0.    0.001 0.025 0.    0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.023 0.    0.    0.    0.   ]
 [0.679 0.    0.    0.119 0.    0.    0.    0.    0.   ]
 [0.304 0.    0.    0.    0.    0.    0.    0.    0.   ]
 [0.    0.023 0.    0.    0.    0.898 0.    0.    0.   ]
 [0.    0.    0.001 0.    0.    0.    0.896 0.    0.   ]
 [0.    0.    0.    0.    0.    0.001 0.    0.974 0.   ]
 [0.    0.    0.797 0.178 0.    0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.921 0.    0.079 0.    0.   ]]
```

For illustration, this solution can be represented with edge width proportional
to the weight of the solution.<br>

<img src="https://q-berthet.github.io/pictures/graph-soft-before.png" width=500>

We consider an example of scalar function on this solution, here the weight of
the perturbed solution on the edge from node 6 to 8 (of current value `0.079`).

```python
def i_to_j_weight_fn(i, j, paths):
 return paths[..., i, j]
with tf.GradientTape() as tape:
  cs = tf.Variable(final_edges_costs)
  weights = edge_costs_to_weights(cs)
  pert_paths = perturbed_shortest_path(weights)
  i_to_j_weight = pert_paths[..., 8, 6]
grad, = tape.gradient(i_to_j_weight, [cs])
```

This provides a direction in which to modify the vector of four edge costs, to
increase the weight on this solution, obtained thanks to our perturbed version
of the optimizer.

```python
>>> grad
[-2.0993764,  2.076386 ,  2.042395 ,  2.0411625]
```

Running gradient *ascent* for 30 steps on this vector of four edge costs to
*increase* the weight of the edge from 6 to 8 modifies the problem. Its new
perturbed solution has a corresponding edge weight of `0.989`. The new problem
and its perturbed solution can be vizualized as follows.<br>

<img src="https://q-berthet.github.io/pictures/graph-soft-after.png" width=500>

## References

Berthet Q., Blondel M., Teboul O., Cuturi M., Vert J.-P., Bach F.,
[Learning with Differentiable Perturbed Optimizers](https://arxiv.org/abs/2002.08676),
NeurIPS 2020

## License

Licensed under the
[Apache 2.0](https://github.com/google-research/google-research/blob/master/LICENSE)
License.

## Disclaimer

This is not an official Google product.
