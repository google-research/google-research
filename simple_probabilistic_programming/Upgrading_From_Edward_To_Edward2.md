# Upgrading from Edward to Edward2

This guide outlines how to port code from the
[Edward](http://edwardlib.org/)
probabilistic programming system to
[Edward2](https://github.com/google-research/google-research/tree/master/simple_probabilistic_programming).
We recommend Edward users use Edward2 for specifying models and other TensorFlow
Probability primitives for performing downstream computation.

Edward2 is a distillation of Edward. It is a low-level language for specifying
probabilistic models as programs and manipulating their computation.
Probabilistic inference, criticism, and any other part of the scientific process
(Box, 1976) use arbitrary TensorFlow ops. Their associated abstractions live in
the TensorFlow ecosystem and do not strictly require Edward2.

For examples:

+ Probabilistic PCA
  ([Edward](https://github.com/blei-lab/edward/blob/master/notebooks/probabilistic_pca.ipynb),
  [TensorFlow Probability](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_PCA.ipynb))
+ Eight schools
  ([Edward](https://github.com/blei-lab/edward/blob/master/notebooks/eight_schools.ipynb),
  [TensorFlow Probability](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Eight_Schools.ipynb))
+ Linear mixed effects models
  ([Edward](https://github.com/blei-lab/edward/blob/master/notebooks/linear_mixed_effects_models.ipynb),
  [TensorFlow Probability](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Linear_Mixed_Effects_Models.ipynb))
+ Variational Autoencoder
  ([Edward](https://github.com/blei-lab/edward/blob/master/examples/vae_convolutional.py),
  [TensorFlow Probability](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/vae.py))
+ Deep Exponential Family
  ([Edward](https://github.com/blei-lab/edward/blob/master/examples/deep_exponential_family.py),
  [TensorFlow Probability](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/deep_exponential_family.py))
+ Mixture of Gaussians
  ([Edward](https://github.com/blei-lab/edward/blob/master/notebooks/unsupervised.ipynb),
  [TensorFlow Probability](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Bayesian_Gaussian_Mixture_Model.ipynb))
+ Logistic regression
  ([Edward](https://github.com/blei-lab/edward/blob/master/examples/bayesian_logistic_regression.py),
  [TensorFlow Probability](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/logistic_regression.py))

Are you having difficulties upgrading to Edward2? Raise a
[GitHub issue](https://github.com/google-research/google-research/issues)
and we're happy to help. Alternatively, if you have tips, feel free to send a
pull request to improve this guide.

## Namespaces

__Edward__.

```python
import edward as ed
from edward.models import Empirical, Gamma, Poisson

dir(ed)
## ['criticisms',
##  'inferences',
##  'models',
##  'util',
##   ...,  # criticisms in global namespace for convenience
##   ...,  # inference algorithms in global namespace for convenience
##   ...]  # utility functions in global namespace for convenience
```

__Edward2__.

```python
import simple_probabilistic_programming as ed  # install locally from this repo

dir(ed)
## [...,  # random variables
##  'as_random_variable',  # various tools for manipulating program execution
##  'get_tracer',
##  'make_log_joint_fn',
##  'trace']
```

## Probabilistic Models

__Edward__. You write models inline with any other code, composing
random variables. As illustration, consider a
deep exponential family
(Ranganath et al., 2015). (For runnable versions of the example code presented
here, see the full
[Edward](https://github.com/blei-lab/edward/blob/master/examples/deep_exponential_family.py)
and
[Edward2](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/deep_exponential_family.py)
source files.)

```python
bag_of_words = np.random.poisson(5., size=[256, 32000])  # training data as matrix of counts
data_size, feature_size = bag_of_words.shape  # number of documents x words (vocabulary)
units = [100, 30, 15]  # number of stochastic units per layer
shape = 0.1  # Gamma shape parameter

w2 = Gamma(0.1, 0.3, sample_shape=[units[2], units[1]])
w1 = Gamma(0.1, 0.3, sample_shape=[units[1], units[0]])
w0 = Gamma(0.1, 0.3, sample_shape=[units[0], feature_size])

z2 = Gamma(0.1, 0.1, sample_shape=[data_size, units[2]])
z1 = Gamma(shape, shape / tf.matmul(z2, w2))
z0 = Gamma(shape, shape / tf.matmul(z1, w1))
x = Poisson(tf.matmul(z1, w0))
```

__Edward2__. You write models as functions, where
random variables operate with the same behavior as Edward's.

```python
def deep_exponential_family(data_size, feature_size, units, shape):
  """A multi-layered topic model over a documents-by-terms matrix."""
  w2 = ed.Gamma(0.1, 0.3, sample_shape=[units[2], units[1]], name="w2")
  w1 = ed.Gamma(0.1, 0.3, sample_shape=[units[1], units[0]], name="w1")
  w0 = ed.Gamma(0.1, 0.3, sample_shape=[units[0], feature_size], name="w0")

  z2 = ed.Gamma(0.1, 0.1, sample_shape=[data_size, units[2]], name="z2")
  z1 = ed.Gamma(shape, shape / tf.matmul(z2, w2), name="z1")
  z0 = ed.Gamma(shape, shape / tf.matmul(z1, w1), name="z0")
  x = ed.Poisson(tf.matmul(z0, w0), name="x")
  return x
```

Broadly, the function's outputs capture what the probabilistic program is over
(the `y` in `p(y | x)`), and the function's inputs capture what the
probabilistic program conditions on (the `x` in `p(y | x)`). Note it's best
practice to write names to all random variables: this is useful for cleaner
TensorFlow name scopes as well as for manipulating model computation.

## TensorFlow Sessions

__Edward__. In graph mode, you fetch values from the TensorFlow graph using a
built-in Edward session. Eager mode is not available.

```python
# Generate from model: return np.ndarray of shape (data_size, feature_size).
with ed.get_session() as sess:
  sess.run(x)
```

__Edward2__. In graph mode, you fetch values from the
TensorFlow graph using a TensorFlow session.

```python
# Generate from model: return np.ndarray of shape (data_size, feature_size).
x = deep_exponential_family(data_size, feature_size, units, shape)

with tf.Session() as sess:  # or, e.g., tf.train.MonitoredSession()
  sess.run(x)
```

You can also use Edward2 in eager mode (`tf.enable_eager_execution()`), where
`x` already fetches the sampled NumPy array (obtainable as `x.numpy()`).

## Probabilistic Inference

In Edward, there is a taxonomy of inference algorithms, with many built-in from
the abstract classes of `ed.MonteCarlo` (sampling) and `ed.VariationalInference`
(optimization). In Edward2, inference algorithms are modularized
so that they can depend on arbitrary TensorFlow ops; any associated abstractions
do not live in Edward2. Below we outline variational inference, Markov
chain Monte Carlo, and how to schedule training.

### Variational Inference

__Edward__. You construct random variables with free parameters, representing
the model's posterior approximation. You align these random variables together
with the model's and construct an inference class.

```python
def trainable_positive_pointmass(shape, name=None):
  """Learnable point mass distribution over positive reals."""
  with tf.variable_scope(None, default_name="trainable_positive_pointmass"):
    return PointMass(tf.nn.softplus(tf.get_variable("mean", shape)), name=name)

def trainable_gamma(shape, name=None):
  """Learnable Gamma via shape and scale parameterization."""
  with tf.variable_scope(None, default_name="trainable_gamma"):
    return Gamma(tf.nn.softplus(tf.get_variable("shape", shape)),
                 1.0 / tf.nn.softplus(tf.get_variable("scale", shape)),
                 name=name)

qw2 = trainable_positive_pointmass(w2.shape)
qw1 = trainable_positive_pointmass(w1.shape)
qw0 = trainable_positive_pointmass(w0.shape)
qz2 = trainable_gamma(z2.shape)
qz1 = trainable_gamma(z1.shape)
qz0 = trainable_gamma(z0.shape)

inference = ed.KLqp({w0: qw0, w1: qw1, w2: qw2, z0: qz0, z1: qz1, z2: qz2},
                    data={x: bag_of_words})
```

__Edward2__. We're in the process of making
variational inference easier. For now, you set up variational inference manually
and/or build your own abstractions.

Below we use Edward2's
[tracing](https://github.com/google-research/google-research/blob/master/simple_probabilistic_programming/trace.py)
in order to manipulate model computation. We define the variational
approximation—another Edward2 program—and apply tracers to write the
evidence lower bound (Hinton & Camp, 1993; Jordan, Ghahramani, Jaakkola, & Saul,
1999; Waterhouse, MacKay, & Robinson, 1996).

```python
def deep_exponential_family_variational():
  """Posterior approx. for deep exponential family p(w{0,1,2}, z{1,2,3} | x)."""
  qw2 = trainable_positive_pointmass(w2.shape, name="qw2")  # same func as above but with ed2 rv's
  qw1 = trainable_positive_pointmass(w1.shape, name="qw1")
  qw0 = trainable_positive_pointmass(w0.shape, name="qw0")
  qz2 = trainable_gamma(z2.shape, name="qz2")  # same func as above but with ed2 rv's
  qz1 = trainable_gamma(z1.shape, name="qz1")
  qz0 = trainable_gamma(z0.shape, name="qz0")
  return qw2, qw1, qw0, qz2, qz1, qz0

def make_value_setter(**model_kwargs):
  """Creates a value-setting tracer."""
  def set_values(f, *args, **kwargs):
    """Sets random variable values to its aligned value."""
    name = kwargs.get("name")
    if name in model_kwargs:
      kwargs["value"] = model_kwargs[name]
    return ed.traceable(f)(*args, **kwargs)
  return set_values

# Compute expected log-likelihood. First, sample from the variational
# distribution; second, compute the log-likelihood given the sample.
qw2, qw1, qw0, qz2, qz1, qz0 = deep_exponential_family_variational()

with ed.tape() as model_tape:
  with ed.trace(make_value_setter(w2=qw2, w1=qw1, w0=qw0,
                                  z2=qz2, z1=qz1, z0=qz0)):
    posterior_predictive = deep_exponential_family(data_size, feature_size, units, shape)

log_likelihood = posterior_predictive.distribution.log_prob(bag_of_words)

# Compute analytic KL-divergence between variational and prior distributions.
kl = 0.
for rv_name, variational_rv in [("z0", qz0), ("z1", qz1), ("z2", qz2),
                                ("w0", qw0), ("w1", qw1), ("w2", qw2)]:
  kl += tf.reduce_sum(variational_rv.distribution.kl_divergence(
      model_tape[rv_name].distribution))

elbo = tf.reduce_mean(log_likelihood - kl)
tf.summary.scalar("elbo", elbo)
optimizer = tf.train.AdamOptimizer(1e-3)
train_op = optimizer.minimize(-elbo)
```

### Markov chain Monte Carlo

__Edward__. Similar to variational inference, you construct random variables
with free parameters, representing the model's posterior approximation. You
align these random variables together with the model's and construct an
inference class.

```python
num_samples = 10000  # number of events to approximate posterior

qw2 = Empirical(tf.get_variable("qw2/params", [num_samples, units[2], units[1]]))
qw1 = Empirical(tf.get_variable("qw1/params", [num_samples, units[1], units[0]]))
qw0 = Empirical(tf.get_variable("qw0/params", [num_samples, units[0], feature_size]))
qz2 = Empirical(tf.get_variable("qz2/params", [num_samples, data_size, units[2]]))
qz1 = Empirical(tf.get_variable("qz1/params", [num_samples, data_size, units[1]]))
qz0 = Empirical(tf.get_variable("qz0/params", [num_samples, data_size, units[0]]))

inference = ed.HMC({w0: qw0, w1: qw1, w2: qw2, z0: qz0, z1: qz1, z2: qz2},
                   data={x: bag_of_words})
```

__Edward2__. Use, e.g., the `tfp.mcmc` module. Operating with
`tfp.mcmc` comprises two stages: set up a transition kernel which determines how
one state propagates to the next; and apply the transition kernel over multiple
iterations until convergence.

Below we first rewrite the Edward2 model in terms of its target log-probability
as a function of latent variables. Namely, it is the model's log-joint
probability function with fixed hyperparameters and observations anchored at the
data. We then apply the higher-level `tfp.mcmc.sample_chain` which applies a
Hamiltonian Monte Carlo transition kernel to return a collection of state
transitions.

```python
num_samples = 10000  # number of events to approximate posterior
qw2 = tf.nn.softplus(tf.random_normal([units[2], units[1]]))  # initial state
qw1 = tf.nn.softplus(tf.random_normal([units[1], units[0]]))
qw0 = tf.nn.softplus(tf.random_normal([units[0], feature_size]))
qz2 = tf.nn.softplus(tf.random_normal([data_size, units[2]]))
qz1 = tf.nn.softplus(tf.random_normal([data_size, units[1]]))
qz0 = tf.nn.softplus(tf.random_normal([data_size, units[0]]))

log_joint = ed.make_log_joint_fn(deep_exponential_family)

def target_log_prob_fn(w2, w1, w0, z2, z1, z0):
  """Target log-probability as a function of states."""
  return log_joint(data_size, feature_size, units, shape,
                   w2=w2, w1=w1, w0=w0, z2=z2, z1=z1, z0=z0, x=bag_of_words)

hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    step_size=0.01,
    num_leapfrog_steps=5)
states, kernels_results = tfp.mcmc.sample_chain(
    num_results=num_samples,
    current_state=[qw2, qw1, qw0, qz2, qz1, qz0],
    kernel=hmc_kernel,
    num_burnin_steps=1000)
```

### The Training Loop

__Edward__. To schedule training, you call `inference.run()` which automatically
handles the schedule. Alternatively, you manually schedule training with
`inference`'s class methods.

```python
inference.initialize(n_iter=10000)

tf.global_variables_initializer().run()
for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

inference.finalize()
```

__Edward2__. To schedule training, you use TensorFlow
ops. For an equivalent `inference.run()`-like API, see
[TensorFlow Estimator](https://www.tensorflow.org/guide/estimators)
([example](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/latent_dirichlet_allocation.py)).
For finetuning variational inference, below is one example.

```python
max_steps = 10000  # number of training iterations
model_dir = None  # directory for model checkpoints

sess = tf.Session()
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
start_time = time.time()

sess.run(tf.global_variables_initializer())
for step in range(max_steps):
  start_time = time.time()
  _, elbo_value = sess.run([train_op, elbo])
  if step % 500 == 0:
    duration = time.time() - start_time
    print("Step: {:>3d} Loss: {:.3f} ({:.3f} sec)".format(
        step, elbo_value, duration))
    summary_str = sess.run(summary)
    summary_writer.add_summary(summary_str, step)
    summary_writer.flush()
```

See the
[deep exponential family](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/deep_exponential_family.py)
example for more details.

Finetuning Markov chain Monte Carlo is similar. Instead of tracking a loss
function, one uses, for example, a counter for the number of accepted samples.
This lets us monitor a running statistic of MCMC's acceptance rate by
accumulating `kernel_results.is_accepted` over session runs.

## Model & Inference Criticism

__Edward__. You typically use two functions: `ed.evaluate` for assessing how
model predictions match the true data; and `ed.ppc` for
assessing how data generated from the model matches the true data.

```python
# Build posterior predictive: it is parameterized by a variational posterior sample.
posterior_predictive = ed.copy(
    x, {w0: qw0, w1: qw1, w2: qw2, z0: qz0, z1: qz1, z2: qz2})

# Evaluate average log-likelihood of data.
ed.evaluate('log_likelihood', data={posterior_predictive: bag_of_words})
## np.ndarray of shape ()

# Compare TF-IDF on real vs generated data.
def tfidf(bag_of_words):
  """Computes term-frequency inverse-document-frequency."""
  num_documents = bag_of_words.shape[0]
  idf = tf.log(num_documents) - tf.log(tf.count_nonzero(bag_of_words, axis=0))
  return bag_of_words * idf

observed_statistics, replicated_statistics = ed.ppc(
    lambda data, latent_vars: tf_idx(data[posterior_predictive]),
    {posterior_predictive: bag_of_words},
    n_samples=100)
```

__Edward2__. Build the metric manually or use TensorFlow
abstractions such as `tf.metrics`.

```python
# See posterior_predictive built in Variational Inference section.
log_likelihood = tf.reduce_mean(posterior_predictive.log_prob(bag_of_words))
## tf.Tensor of shape ()

# Simple version: Compare statistics by sampling from model in a for loop.
observed_statistic = sess.run(tfidf(bag_of_words))
replicated_statistic = tfidf(posterior_predictive)
replicated_statistics = [sess.run(replicated_statistic) for _ in range(100)]
```

See
[Bayesian neural networks](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py)
for training with `tf.metrics.accuracy` and
[Eight Schools](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Eight_Schools.ipynb)
for visualizing manually written predictive checks.

## References

1. George Edward Pelham Box. Science and statistics. _Journal of the American Statistical Association_, 71(356), 791–799, 1976.
2. Hinton, G. E., & Camp, D. van. (1993). Keeping the neural networks simple by minimizing the description length of the weights. In Conference on learning theory. ACM.
3. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine Learning, 37(2), 183–233.
4. Rajesh Ranganath, Linpeng Tang, Laurent Charlin, David M. Blei. Deep exponential families. In _Artificial Intelligence and Statistics_, 2015.
5. Waterhouse, S., MacKay, D., & Robinson, T. (1996). Bayesian methods for mixtures of experts. Advances in Neural Information Processing Systems, 351–357.
