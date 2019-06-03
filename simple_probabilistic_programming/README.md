# Simple, Distributed, and Accelerated Probabilistic Programming

This directory contains reference code for the NIPS 2018 paper,
["Simple, Distributed, and Accelerated Probabilistic Programming"](https://arxiv.org/abs/1811.02091).
It's organized as follows:

* [`examples/`](https://github.com/google-research/google-research/blob/master/simple_probabilistic_programming/examples):
  Examples, including an implementation of the No-U-Turn Sampler.
* [`notebooks/`](https://github.com/google-research/google-research/blob/master/simple_probabilistic_programming/notebooks):
  Jupyter notebooks, including a companion notebook for the paper's examples.
* [`*.py`](https://github.com/google-research/google-research/blob/master/simple_probabilistic_programming/):
  Edward2, in its core implementation. It features two backends:
  [`numpy/`](https://github.com/google-research/google-research/blob/master/simple_probabilistic_programming/numpy)
  and
  [`tensorflow/`](https://github.com/google-research/google-research/blob/master/simple_probabilistic_programming/tensorflow).

The implementation, Edward2, is a probabilistic programming language in Python.
It extends the NumPy or TensorFlow ecosystem so that one can declare models as
probabilistic programs and manipulate a model's computation for flexible
training, latent variable inference, and predictions.

Are you upgrading from Edward? Check out the guide
[`Upgrading_from_Edward_to_Edward2.md`](https://github.com/google-research/google-research/blob/master/simple_probabilistic_programming/Upgrading_From_Edward_To_Edward2.md).

## Installation

To install the latest stable version, run

```sh
pip install edward2
```

Edward2 supports two backends: TensorFlow (the default) and
NumPy ([see below to activate](#using-the-numpy-backend)). Installing
`edward2` does not automatically install or update TensorFlow or NumPy. We
recommend installing TensorFlow via `pip install edward2[tensorflow]` or
`pip install edward2[tensorflow_gpu]`. See TensorFlow’s
[installation instructions for details](https://www.tensorflow.org/install/).
You may need to use TensorFlow's nightly package (`tf-nightly`). Alternatively,
install NumPy via `pip install edward2[numpy]`.

## 1. Models as Probabilistic Programs

### Random Variables

In Edward2, we use
[`RandomVariables`](https://github.com/google-research/google-research/blob/master/simple_probabilistic_programming/random_variable.py)
to specify a probabilistic model's structure.
A random variable `rv` carries a probability distribution (`rv.distribution`),
which is a TensorFlow Distribution instance governing the random variable's methods
such as `log_prob` and `sample`.

Random variables are formed like TensorFlow Distributions.

```python
import edward2 as ed

normal_rv = ed.Normal(loc=0., scale=1.)
## <ed.RandomVariable 'Normal/' shape=() dtype=float32>
normal_rv.distribution.log_prob(1.231)
## <tf.Tensor 'Normal/log_prob/sub:0' shape=() dtype=float32>

dirichlet_rv = ed.Dirichlet(concentration=tf.ones([2, 10]))
## <ed.RandomVariable 'Dirichlet/' shape=(2, 10) dtype=float32>
```

By default, instantiating a random variable `rv` creates a sampling op to form
the tensor `rv.value ~ rv.distribution.sample()`. The default number of samples
(controllable via the `sample_shape` argument to `rv`) is one, and if the
optional `value` argument is provided, no sampling op is created. Random
variables can interoperate with TensorFlow ops: the TF ops operate on the sample.

```python
x = ed.Normal(loc=tf.zeros(10), scale=tf.ones(10))
y = 5.
x + y, x / y
## (<tf.Tensor 'add:0' shape=(10,) dtype=float32>,
##  <tf.Tensor 'div:0' shape=(10,) dtype=float32>)
tf.tanh(x * y)
## <tf.Tensor 'Tanh:0' shape=(10,) dtype=float32>
x[2]  # 3rd normal rv
## <tf.Tensor 'strided_slice:0' shape=() dtype=float32>
```

### Probabilistic Models

Probabilistic models in Edward2 are expressed as Python functions that
instantiate one or more `RandomVariables`. Typically, the function ("program")
executes the generative process and returns samples. Inputs to the
function can be thought of as values the model conditions on.

Below we write Bayesian logistic regression, where binary outcomes are generated
given features, coefficients, and an intercept. There is a prior over the
coefficients and intercept. Executing the function adds operations to the
TensorFlow graph, and asking for the result node in a TensorFlow session will
sample coefficients and intercept from the prior, and use these samples to
compute the outcomes.

```python
def logistic_regression(features):
  """Bayesian logistic regression p(y | x) = int p(y | x, w, b) p(w, b) dwdb."""
  coeffs = ed.Normal(loc=tf.zeros(features.shape[1]), scale=1., name="coeffs")
  intercept = ed.Normal(loc=0., scale=1., name="intercept")
  outcomes = ed.Bernoulli(
      logits=tf.tensordot(features, coeffs, [[1], [0]]) + intercept,
      name="outcomes")
  return outcomes

num_features = 10
features = tf.random_normal([100, num_features])
outcomes = logistic_regression(features)

# Execute the model program, returning a sample np.ndarray of shape (100,).
with tf.Session() as sess:
  outcomes_ = sess.run(outcomes)
```

Edward2 programs can also represent distributions beyond those which directly
model data. For example, below we write a learnable distribution with the
intention to approximate it to the logistic regression posterior.

```python
def logistic_regression_posterior(num_features):
  """Posterior of Bayesian logistic regression p(w, b | {x, y})."""
  posterior_coeffs = ed.MultivariateNormalTriL(
      loc=tf.get_variable("coeffs_loc", [num_features]),
      scale_tril=tfp.trainable_distributions.tril_with_diag_softplus_and_shift(
          tf.get_variable("coeffs_scale", [num_features*(num_features+1) / 2])),
      name="coeffs_posterior")
  posterior_intercept = ed.Normal(
      loc=tf.get_variable("intercept_loc", []),
      scale=tf.nn.softplus(tf.get_variable("intercept_scale", [])) + 1e-5,
      name="intercept_posterior")
  return coeffs, intercept

coeffs, intercept = logistic_regression_posterior(num_features)

# Execute the program, returning a sample
# (np.ndarray of shape (55,), np.ndarray of shape ()).
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  posterior_coeffs_, posterior_ntercept_ = sess.run(
      [posterior_coeffs, posterior_intercept])
```

## 2. Manipulating Model Computation

### Tracing

Training and testing probabilistic models typically require more than just
samples from the generative process. To enable flexible training and testing, we
manipulate the model's computation using
[tracing](https://github.com/google-research/google-research/blob/master/simple_probabilistic_programming/tracer.py).

A tracer is a function that acts on another function `f` and its arguments
`*args`, `**kwargs`. It performs various computations before returning an output
(typically `f(*args, **kwargs)`: the result of applying the function itself).
The `ed.trace` context manager pushes tracers onto a stack, and any
traceable function is intercepted by the stack. All random variable
constructors are traceable.

Below we trace the logistic regression model's generative process. In
particular, we make predictions with its learned posterior means rather than
with its priors.

```python
def set_prior_to_posterior_mean(f, *args, **kwargs):
  """Forms posterior predictions, setting each prior to its posterior mean."""
  name = kwargs.get("name")
  if name == "coeffs":
    return posterior_coeffs.distribution.mean()
  elif name == "intercept":
    return posterior_intercept.distribution.mean()
  return f(*args, **kwargs)

with ed.trace(set_prior_to_posterior_mean):
  predictions = logistic_regression(features)

training_accuracy = (
    tf.reduce_sum(tf.cast(tf.equal(predictions, outcomes), tf.float32)) /
    tf.cast(tf.shape(outcomes), tf.float32))
```

### Program Transformations

Using tracing, one can also apply program transformations, which map
from one representation of a model to another. This provides convenient access
to different model properties depending on the downstream use case.

For example, Markov chain Monte Carlo algorithms often require a model's
log-joint probability function as input. Below we take the Bayesian logistic
regression program which specifies a generative process, and apply the built-in
`ed.make_log_joint` transformation to obtain its log-joint probability function.
The log-joint function takes as input the generative program's original inputs
as well as random variables in the program. It returns a scalar Tensor
summing over all random variable log-probabilities.

In our example, `features` and `outcomes` are fixed, and we want to use
Hamiltonian Monte Carlo to draw samples from the posterior distribution of
`coeffs` and `intercept`. To this use, we create `target_log_prob_fn`, which
takes just `coeffs` and `intercept` as arguments and pins the input `features`
and output rv `outcomes` to its known values.

```python
from simple_probabilistic_programming.examples import no_u_turn_sampler

tf.enable_eager_execution()

# Set up training data.
features = tf.random_normal([100, 55])
outcomes = tf.random_uniform([100], minval=0, maxval=2, dtype=tf.int32)

# Pass target log-probability function to MCMC transition kernel.
log_joint = ed.make_log_joint_fn(logistic_regression)

def target_log_prob_fn(coeffs, intercept):
  """Target log-probability as a function of states."""
  return log_joint(features,
                   coeffs=coeffs,
                   intercept=intercept,
                   outcomes=outcomes)

coeffs_samples = []
intercept_samples = []
coeffs = tf.random_normal([55])
intercept = tf.random_normal([])
target_log_prob = None
grads_target_log_prob = None
for _ in range(1000):
  [
      [coeffs, intercepts],
      target_log_prob,
      grads_target_log_prob,
  ] = no_u_turn_sampler.kernel(
          target_log_prob_fn=target_log_prob_fn,
          current_state=[coeffs, intercept],
          step_size=[0.1, 0.1],
          current_target_log_prob=target_log_prob,
          current_grads_target_log_prob=grads_target_log_prob)
  coeffs_samples.append(coeffs)
  intercept_samples.append(coeffs)
```

The returned `coeffs_samples` and `intercept_samples` contain 1,000 posterior
samples for `coeffs` and `intercept` respectively. They may be used, for
example, to evaluate the model's posterior predictive on new data.

## Using the NumPy backend

Using alternative backends is as simple as the following:

```python
import edward2.numpy as ed
```

In the NumPy backend, Edward2 wraps SciPy distributions. For example, here's
linear regression.

```python
def linear_regression(features, prior_precision):
  beta = ed.norm.rvs(loc=0.,
                     scale=1. / np.sqrt(prior_precision),
                     size=features.shape[1])
  y = ed.norm.rvs(loc=np.dot(features, beta), scale=1., size=1)
  return y
```

## References

> Tran, D., Hoffman, M. D., Moore, D., Suter, C., Vasudevan S., Radul A.,
> Johnson M., and Saurous R. A. (2018).
> [Simple, Distributed, and Accelerated Probabilistic Programming](https://arxiv.org/abs/1811.02091).
> In _Neural Information Processing Systems_.

```none
@inproceedings{tran2018simple,
  author = {Dustin Tran and Matthew D. Hoffman and Dave Moore and Christopher Suter and Srinivas Vasudevan and Alexey Radul and Matthew Johnson and Rif A. Saurous},
  title = {Simple, Distributed, and Accelerated Probabilistic Programming},
  booktitle = {Neural Information Processing Systems},
  year = {2018},
}
```
