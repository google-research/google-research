# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusion for discrete state spaces."""

from . import utils
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp
import scipy


def make_diffusion(hps, num_bits):
  """HParams -> diffusion object."""
  return CategoricalDiffusion(
      betas=get_diffusion_betas(hps.diffusion_betas),
      model_prediction=hps.model_prediction,
      model_output=hps.args.model_output,
      transition_mat_type=hps.transition_mat_type,
      transition_bands=hps.transition_bands,
      loss_type=hps.loss_type,
      hybrid_coeff=hps.hybrid_coeff,
      num_bits=num_bits)


def get_diffusion_betas(spec):
  """Get betas from the hyperparameters."""
  if spec.type == 'linear':
    # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
    # To be used with Gaussian diffusion models in continuous and discrete
    # state spaces.
    # To be used with transition_mat_type = 'gaussian'
    return onp.linspace(spec.start, spec.stop, spec.num_timesteps)
  elif spec.type == 'cosine':
    # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
    # To be used with transition_mat_type = 'uniform'.
    steps = (
        onp.arange(spec.num_timesteps + 1, dtype=onp.float64) /
        spec.num_timesteps)
    alpha_bar = onp.cos((steps + 0.008) / 1.008 * onp.pi / 2)
    betas = onp.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
    return betas
  elif spec.type == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
    # To be used with absorbing state models.
    # ensures that the probability of decaying to the absorbing state
    # increases linearly over time, and is 1 for t = T-1 (the final time).
    # To be used with transition_mat_type = 'absorbing'
    return 1. / onp.linspace(spec.num_timesteps, 1., spec.num_timesteps)
  else:
    raise NotImplementedError(spec.type)


class CategoricalDiffusion:
  """Discrete state space diffusion process.

  Time convention: noisy data is labeled x_0, ..., x_{T-1}, and original data
  is labeled x_start (or x_{-1}). This convention differs from the papers,
  which use x_1, ..., x_T for noisy data and x_0 for original data.
  """

  def __init__(self, *, betas, model_prediction, model_output,
               transition_mat_type, transition_bands, loss_type, hybrid_coeff,
               num_bits, jax_dtype=jnp.float32):

    self.model_prediction = model_prediction  # x_start, xprev
    self.model_output = model_output  # logits or logistic_pars
    self.loss_type = loss_type  # kl, hybrid, cross_entropy_x_start
    self.hybrid_coeff = hybrid_coeff
    self.jax_dtype = jax_dtype

    self.num_bits = num_bits
    # Data \in {0, ..., num_pixel_vals-1}
    self.num_pixel_vals = 2**self.num_bits
    self.transition_bands = transition_bands
    self.transition_mat_type = transition_mat_type
    self.eps = 1.e-6

    if not isinstance(betas, onp.ndarray):
      raise ValueError('expected betas to be a numpy array')
    if not ((betas > 0).all() and (betas <= 1).all()):
      raise ValueError('betas must be in (0, 1]')

    # Computations here in float64 for accuracy
    self.betas = betas = betas.astype(onp.float64)
    self.num_timesteps, = betas.shape

    # Construct transition matrices for q(x_t|x_{t-1})
    # NOTE: t goes from {0, ..., T-1}
    if self.transition_mat_type == 'uniform':
      q_one_step_mats = [self._get_transition_mat(t)
                         for t in range(0, self.num_timesteps)]
    elif self.transition_mat_type == 'gaussian':
      q_one_step_mats = [self._get_gaussian_transition_mat(t)
                         for t in range(0, self.num_timesteps)]
    elif self.transition_mat_type == 'absorbing':
      q_one_step_mats = [self._get_absorbing_transition_mat(t)
                         for t in range(0, self.num_timesteps)]
    else:
      raise ValueError(
          f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
          f", but is {self.transition_mat_type}"
          )

    self.q_onestep_mats = onp.stack(q_one_step_mats, axis=0)
    assert self.q_onestep_mats.shape == (self.num_timesteps,
                                         self.num_pixel_vals,
                                         self.num_pixel_vals)

    # Construct transition matrices for q(x_t|x_start)
    q_mat_t = self.q_onestep_mats[0]
    q_mats = [q_mat_t]
    for t in range(1, self.num_timesteps):
      # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
      q_mat_t = onp.tensordot(q_mat_t, self.q_onestep_mats[t],
                              axes=[[1], [0]])
      q_mats.append(q_mat_t)
    self.q_mats = onp.stack(q_mats, axis=0)
    assert self.q_mats.shape == (self.num_timesteps, self.num_pixel_vals,
                                 self.num_pixel_vals), self.q_mats.shape

    # Don't precompute transition matrices for q(x_{t-1} | x_t, x_start)
    # Can be computed from self.q_mats and self.q_one_step_mats.
    # Only need transpose of q_onestep_mats for posterior computation.
    self.transpose_q_onestep_mats = onp.transpose(self.q_onestep_mats,
                                                  axes=(0, 2, 1))
    del self.q_onestep_mats

  def _get_full_transition_mat(self, t):
    """Computes transition matrix for q(x_t|x_{t-1}).

    Contrary to the band diagonal version, this method constructs a transition
    matrix with uniform probability to all other states.

    Args:
      t: timestep. integer scalar.

    Returns:
      Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    beta_t = self.betas[t]
    mat = onp.full(shape=(self.num_pixel_vals, self.num_pixel_vals),
                   fill_value=beta_t/float(self.num_pixel_vals),
                   dtype=onp.float64)
    diag_indices = onp.diag_indices_from(mat)
    diag_val = 1. - beta_t * (self.num_pixel_vals-1.)/self.num_pixel_vals
    mat[diag_indices] = diag_val
    return mat

  def _get_transition_mat(self, t):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    This method constructs a transition
    matrix Q with
    Q_{ij} = beta_t / num_pixel_vals       if |i-j| <= self.transition_bands
             1 - \sum_{l \neq i} Q_{il} if i==j.
             0                          else.

    Args:
      t: timestep. integer scalar (or numpy array?)

    Returns:
      Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    if self.transition_bands is None:
      return self._get_full_transition_mat(t)
    # Assumes num_off_diags < num_pixel_vals
    beta_t = self.betas[t]

    mat = onp.zeros((self.num_pixel_vals, self.num_pixel_vals),
                    dtype=onp.float64)
    off_diag = onp.full(shape=(self.num_pixel_vals-1,),
                        fill_value=beta_t/float(self.num_pixel_vals),
                        dtype=onp.float64)
    for k in range(1, self.transition_bands + 1):
      mat += onp.diag(off_diag, k=k)
      mat += onp.diag(off_diag, k=-k)
      off_diag = off_diag[:-1]

    # Add diagonal values such that rows sum to one.
    diag = 1. - mat.sum(1)
    mat += onp.diag(diag, k=0)
    return mat

  def _get_gaussian_transition_mat(self, t):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    This method constructs a transition matrix Q with
    decaying entries as a function of how far off diagonal the entry is.
    Normalization option 1:
    Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
             1 - \sum_{l \neq i} Q_{il}  if i==j.
             0                          else.

    Normalization option 2:
    tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                     0                        else.

    Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

    Args:
      t: timestep. integer scalar (or numpy array?)

    Returns:
      Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    transition_bands = self.transition_bands if self.transition_bands else self.num_pixel_vals - 1

    beta_t = self.betas[t]

    mat = onp.zeros((self.num_pixel_vals, self.num_pixel_vals),
                    dtype=onp.float64)

    # Make the values correspond to a similar type of gaussian as in the
    # gaussian diffusion case for continuous state spaces.
    values = onp.linspace(start=0., stop=255., num=self.num_pixel_vals,
                          endpoint=True, dtype=onp.float64)
    values = values * 2./ (self.num_pixel_vals - 1.)
    values = values[:transition_bands+1]
    values = -values * values / beta_t

    values = onp.concatenate([values[:0:-1], values], axis=0)
    values = scipy.special.softmax(values, axis=0)
    values = values[transition_bands:]
    for k in range(1, transition_bands + 1):
      off_diag = onp.full(shape=(self.num_pixel_vals - k,),
                          fill_value=values[k],
                          dtype=onp.float64)

      mat += onp.diag(off_diag, k=k)
      mat += onp.diag(off_diag, k=-k)

    # Add diagonal values such that rows and columns sum to one.
    # Technically only the ROWS need to sum to one
    # NOTE: this normalization leads to a doubly stochastic matrix,
    # which is necessary if we want to have a uniform stationary distribution.
    diag = 1. - mat.sum(1)
    mat += onp.diag(diag, k=0)

    return mat

  def _get_absorbing_transition_mat(self, t):
    """Computes transition matrix for q(x_t|x_{t-1}).

    Has an absorbing state for pixelvalues self.num_pixel_vals//2.

    Args:
      t: timestep. integer scalar.

    Returns:
      Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    beta_t = self.betas[t]

    diag = onp.full(shape=(self.num_pixel_vals,), fill_value=1. - beta_t,
                    dtype=onp.float64)
    mat = onp.diag(diag, k=0)
    # Add beta_t to the num_pixel_vals/2-th column for the absorbing state.
    mat[:, self.num_pixel_vals//2] += beta_t

    return mat

  def _at(self, a, t, x):
    """Extract coefficients at specified timesteps t and conditioning data x.

    Args:
      a: np.ndarray: plain NumPy float64 array of constants indexed by time.
      t: jnp.ndarray: Jax array of time indices, shape = (batch_size,).
      x: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
        (Noisy) data. Should not be of one hot representation, but have integer
        values representing the class values.

    Returns:
      a[t, x]: jnp.ndarray: Jax array.
    """
    a = jnp.asarray(a, dtype=self.jax_dtype)
    t_broadcast = jnp.expand_dims(t, tuple(range(1, x.ndim)))

    # x.shape = (bs, height, width, channels)
    # t_broadcast_shape = (bs, 1, 1, 1)
    # a.shape = (num_timesteps, num_pixel_vals, num_pixel_vals)
    # out.shape = (bs, height, width, channels, num_pixel_vals)
    # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
    return a[t_broadcast, x]

  def _at_onehot(self, a, t, x):
    """Extract coefficients at specified timesteps t and conditioning data x.

    Args:
      a: np.ndarray: plain NumPy float64 array of constants indexed by time.
      t: jnp.ndarray: Jax array of time indices, shape = (bs,).
      x: jnp.ndarray: jax array, shape (bs, ..., num_pixel_vals), float32 type.
        (Noisy) data. Should be of one-hot-type representation.

    Returns:
      out: jnp.ndarray: Jax array. output of dot(x, a[t], axis=[[-1], [1]]).
        shape = (bs, ..., num_pixel_vals)
    """
    a = jnp.asarray(a, dtype=self.jax_dtype)

    # x.shape = (bs, height, width, channels, num_pixel_vals)
    # a[t]shape = (bs, num_pixel_vals, num_pixel_vals)
    # out.shape = (bs, height, width, channels, num_pixel_vals)
    return jnp.matmul(x, a[t, None, None, Ellipsis],
                      precision=jax.lax.Precision.HIGHEST)

  def q_probs(self, x_start, t):
    """Compute probabilities of q(x_t | x_start).

    Args:
      x_start: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
         Should not be of one hot representation, but have integer values
         representing the class values.
      t: jnp.ndarray: jax array of shape (bs,).

    Returns:
      probs: jnp.ndarray: jax array, shape (bs, x_start.shape[1:],
                                            num_pixel_vals).
    """
    return self._at(self.q_mats, t, x_start)

  def q_sample(self, x_start, t, noise):
    """Sample from q(x_t | x_start) (i.e. add noise to the data).

    Args:
      x_start: jnp.array: original clean data, in integer form (not onehot).
        shape = (bs, ...).
      t: :jnp.array: timestep of the diffusion process, shape (bs,).
      noise: jnp.ndarray: uniform noise on [0, 1) used to sample noisy data.
        Should be of shape (*x_start.shape, num_pixel_vals).

    Returns:
      sample: jnp.ndarray: same shape as x_start. noisy data.
    """
    assert noise.shape == x_start.shape + (self.num_pixel_vals,)
    logits = jnp.log(self.q_probs(x_start, t) + self.eps)

    # To avoid numerical issues clip the noise to a minimum value
    noise = jnp.clip(noise, a_min=jnp.finfo(noise.dtype).tiny, a_max=1.)
    gumbel_noise = - jnp.log(-jnp.log(noise))
    return jnp.argmax(logits + gumbel_noise, axis=-1)

  def _get_logits_from_logistic_pars(self, loc, log_scale):
    """Computes logits for an underlying logistic distribution."""

    loc = jnp.expand_dims(loc, axis=-1)
    log_scale = jnp.expand_dims(log_scale, axis=-1)

    # Shift log_scale such that if it's zero the probs have a scale
    # that is not too wide and not too narrow either.
    inv_scale = jnp.exp(- (log_scale - 2.))

    bin_width = 2. / (self.num_pixel_vals - 1.)
    bin_centers = jnp.linspace(start=-1., stop=1., num=self.num_pixel_vals,
                               endpoint=True)

    bin_centers = jnp.expand_dims(bin_centers,
                                  axis=tuple(range(0, loc.ndim-1)))

    bin_centers = bin_centers - loc
    log_cdf_min = jax.nn.log_sigmoid(
        inv_scale * (bin_centers - 0.5 * bin_width))
    log_cdf_plus = jax.nn.log_sigmoid(
        inv_scale * (bin_centers + 0.5 * bin_width))

    logits = utils.log_min_exp(log_cdf_plus, log_cdf_min, self.eps)

    # Normalization:
    # # Option 1:
    # # Assign cdf over range (-\inf, x + 0.5] to pmf for pixel with
    # # value x = 0.
    # logits = logits.at[..., 0].set(log_cdf_plus[..., 0])
    # # Assign cdf over range (x - 0.5, \inf) to pmf for pixel with
    # # value x = 255.
    # log_one_minus_cdf_min = - jax.nn.softplus(
    #     inv_scale * (bin_centers - 0.5 * bin_width))
    # logits = logits.at[..., -1].set(log_one_minus_cdf_min[..., -1])
    # # Option 2:
    # # Alternatively normalize by reweighting all terms. This avoids
    # # sharp peaks at 0 and 255.
    # since we are outputting logits here, we don't need to do anything.
    # they will be normalized by softmax anyway.

    return logits

  def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
    """Compute logits of q(x_{t-1} | x_t, x_start)."""

    if x_start_logits:
      assert x_start.shape == x_t.shape + (self.num_pixel_vals,), (
          x_start.shape, x_t.shape)
    else:
      assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

    fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
    if x_start_logits:
      fact2 = self._at_onehot(self.q_mats, t-1,
                              jax.nn.softmax(x_start, axis=-1))
      tzero_logits = x_start
    else:
      fact2 = self._at(self.q_mats, t-1, x_start)
      tzero_logits = jnp.log(
          jax.nn.one_hot(x_start, num_classes=self.num_pixel_vals)
          + self.eps)

    # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
    # where x_{-1} == x_start. This should be equal the log of x_0.
    out = jnp.log(fact1 + self.eps) + jnp.log(fact2 + self.eps)
    t_broadcast = jnp.expand_dims(t, tuple(range(1, out.ndim)))
    return jnp.where(t_broadcast == 0, tzero_logits,
                     out)

  def p_logits(self, model_fn, *, x, t):
    """Compute logits of p(x_{t-1} | x_t)."""
    assert t.shape == (x.shape[0],)
    model_output = model_fn(x, t)

    if self.model_output == 'logits':
      model_logits = model_output

    elif self.model_output == 'logistic_pars':
      # Get logits out of discretized logistic distribution.
      loc, log_scale = model_output
      model_logits = self._get_logits_from_logistic_pars(loc, log_scale)

    else:
      raise NotImplementedError(self.model_output)

    if self.model_prediction == 'x_start':
      # Predict the logits of p(x_{t-1}|x_t) by parameterizing this distribution
      # as ~ sum_{pred_x_start} q(x_{t-1}, x_t |pred_x_start)p(pred_x_start|x_t)
      pred_x_start_logits = model_logits

      t_broadcast = jnp.expand_dims(t, tuple(range(1, model_logits.ndim)))
      model_logits = jnp.where(t_broadcast == 0,
                               pred_x_start_logits,
                               self.q_posterior_logits(pred_x_start_logits, x,
                                                       t, x_start_logits=True)
                               )

    elif self.model_prediction == 'xprev':
      # Use the logits out of the model directly as the logits for
      # p(x_{t-1}|x_t). model_logits are already set correctly.
      # NOTE: the pred_x_start_logits in this case makes no sense.
      # For Gaussian DDPM diffusion the model predicts the mean of
      # p(x_{t-1}}|x_t), and uses inserts this as the eq for the mean of
      # q(x_{t-1}}|x_t, x_0) to compute the predicted x_0/x_start.
      # The equivalent for the categorical case is nontrivial.
      pred_x_start_logits = model_logits
      raise NotImplementedError(self.model_prediction)

    assert (model_logits.shape ==
            pred_x_start_logits.shape == x.shape + (self.num_pixel_vals,))
    return model_logits, pred_x_start_logits

  # === Sampling ===

  def p_sample(self, model_fn, *, x, t, noise):
    """Sample one timestep from the model p(x_{t-1} | x_t)."""
    model_logits, pred_x_start_logits = self.p_logits(
        model_fn=model_fn, x=x, t=t)
    assert noise.shape == model_logits.shape, noise.shape

    # No noise when t == 0
    # NOTE: for t=0 this just "samples" from the argmax
    #   as opposed to "sampling" from the mean in the gaussian case.
    nonzero_mask = (t != 0).astype(x.dtype).reshape(x.shape[0],
                                                    *([1] * (len(x.shape))))
    # For numerical precision clip the noise to a minimum value
    noise = jnp.clip(noise, a_min=jnp.finfo(noise.dtype).tiny, a_max=1.)
    gumbel_noise = -jnp.log(-jnp.log(noise))

    sample = jnp.argmax(model_logits + nonzero_mask * gumbel_noise, axis=-1)

    assert sample.shape == x.shape
    assert pred_x_start_logits.shape == model_logits.shape
    return sample, jax.nn.softmax(pred_x_start_logits, axis=-1)

  def p_sample_loop(self, model_fn, *, shape, rng,
                    num_timesteps=None, return_x_init=False):
    """Ancestral sampling."""
    init_rng, body_rng = jax.random.split(rng)
    del rng

    noise_shape = shape + (self.num_pixel_vals,)
    def body_fun(i, x):
      t = jnp.full([shape[0]], self.num_timesteps - 1 - i)
      x, _ = self.p_sample(
          model_fn=model_fn,
          x=x,
          t=t,
          noise=jax.random.uniform(
              jax.random.fold_in(body_rng, i), shape=noise_shape)
          )
      return x

    if self.transition_mat_type in ['gaussian', 'uniform']:
      # Stationary distribution is a uniform distribution over all pixel values.
      x_init = jax.random.randint(init_rng, shape=shape,
                                  minval=0, maxval=self.num_pixel_vals)

    elif self.transition_mat_type == 'absorbing':
      # Stationary distribution is a kronecker delta distribution
      # with all its mass on the absorbing state.
      # Absorbing state is located at rgb values (128, 128, 128)
      x_init = jnp.full(shape=shape, fill_value=self.num_pixel_vals//2,
                        dtype=jnp.int32)
    else:
      raise ValueError(
          f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
          f", but is {self.transition_mat_type}"
          )

    del init_rng

    if num_timesteps is None:
      num_timesteps = self.num_timesteps

    final_x = lax.fori_loop(lower=0, upper=num_timesteps,
                            body_fun=body_fun, init_val=x_init)
    assert final_x.shape == shape
    if return_x_init:
      return x_init, final_x
    else:
      return final_x

  # === Log likelihood / loss calculation ===

  def vb_terms_bpd(self, model_fn, *, x_start, x_t, t):
    """Calculate specified terms of the variational bound.

    Args:
      model_fn: the denoising network
      x_start: original clean data
      x_t: noisy data
      t: timestep of the noisy data (and the corresponding term of the bound
        to return)

    Returns:
      a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
      (specified by `t`), and `pred_x_start_logits` is logits of
      the denoised image.
    """
    true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_logits=False)
    model_logits, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t)

    kl = utils.categorical_kl_logits(logits1=true_logits, logits2=model_logits)
    assert kl.shape == x_start.shape
    kl = utils.meanflat(kl) / onp.log(2.)

    decoder_nll = -utils.categorical_log_likelihood(x_start, model_logits)
    assert decoder_nll.shape == x_start.shape
    decoder_nll = utils.meanflat(decoder_nll) / onp.log(2.)

    # At the first timestep return the decoder NLL,
    # otherwise return KL(q(x_{t-1}|x_t,x_start) || p(x_{t-1}|x_t))
    assert kl.shape == decoder_nll.shape == t.shape == (x_start.shape[0],)
    return jnp.where(t == 0, decoder_nll, kl), pred_x_start_logits

  def prior_bpd(self, x_start):
    """KL(q(x_{T-1}|x_start)|| U(x_{T-1}|0, num_pixel_vals-1))."""
    q_probs = self.q_probs(
        x_start=x_start,
        t=jnp.full((x_start.shape[0],), self.num_timesteps - 1))

    if self.transition_mat_type in ['gaussian', 'uniform']:
      # Stationary distribution is a uniform distribution over all pixel values.
      prior_probs = jnp.ones_like(q_probs) / self.num_pixel_vals

    elif self.transition_mat_type == 'absorbing':
      # Stationary distribution is a kronecker delta distribution
      # with all its mass on the absorbing state.
      # Absorbing state is located at rgb values (128, 128, 128)
      absorbing_int = jnp.full(shape=q_probs.shape[:-1],
                               fill_value=self.num_pixel_vals//2,
                               dtype=jnp.int32)
      prior_probs = jax.nn.one_hot(absorbing_int,
                                   num_classes=self.num_pixel_vals,
                                   axis=-1, dtype=self.jax_dtype)
    else:
      raise ValueError(
          f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
          f", but is {self.transition_mat_type}"
          )

    assert prior_probs.shape == q_probs.shape

    kl_prior = utils.categorical_kl_probs(
        q_probs, prior_probs)
    assert kl_prior.shape == x_start.shape
    return utils.meanflat(kl_prior) / onp.log(2.)

  def cross_entropy_x_start(self, x_start, pred_x_start_logits):
    """Calculate crossentropy between x_start and predicted x_start.

    Args:
      x_start: original clean data
      pred_x_start_logits: predicted_logits

    Returns:
      ce: cross entropy.
    """

    ce = -utils.categorical_log_likelihood(x_start, pred_x_start_logits)
    assert ce.shape == x_start.shape
    ce = utils.meanflat(ce) / onp.log(2.)

    assert ce.shape == (x_start.shape[0],)

    return ce

  def training_losses(self, model_fn, *, x_start, rng):
    """Training loss calculation."""

    # Add noise to data
    noise_rng, time_rng = jax.random.split(rng)
    noise = jax.random.uniform(noise_rng,
                               shape=x_start.shape + (self.num_pixel_vals,))
    t = jax.random.randint(time_rng, shape=(x_start.shape[0],), minval=0,
                           maxval=self.num_timesteps, dtype=jnp.int32)

    # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint
    # itself.
    x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

    # Calculate the loss
    if self.loss_type == 'kl':
      # Optimizes the variational bound L_vb.
      losses, _ = self.vb_terms_bpd(
          model_fn=model_fn, x_start=x_start, x_t=x_t, t=t)

    elif self.loss_type == 'cross_entropy_x_start':
      # Optimizes - sum_x_start x_start log pred_x_start.
      _, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t)
      losses = self.cross_entropy_x_start(
          x_start=x_start, pred_x_start_logits=pred_x_start_logits)

    elif self.loss_type == 'hybrid':
      # Optimizes L_vb - lambda * sum_x_start x_start log pred_x_start.
      vb_losses, pred_x_start_logits = self.vb_terms_bpd(
          model_fn=model_fn, x_start=x_start, x_t=x_t, t=t)
      ce_losses = self.cross_entropy_x_start(
          x_start=x_start, pred_x_start_logits=pred_x_start_logits)
      losses = vb_losses + self.hybrid_coeff * ce_losses

    else:
      raise NotImplementedError(self.loss_type)

    assert losses.shape == t.shape
    return losses

  def calc_bpd_loop(self, model_fn, *, x_start, rng):
    """Calculate variational bound (loop over all timesteps and sum)."""
    batch_size = x_start.shape[0]

    noise_shape = x_start.shape + (self.num_pixel_vals,)
    def map_fn(map_val):
      t, cur_rng = map_val
      # Calculate VB term at the current timestep
      t = jnp.full((batch_size,), t)
      vb, _ = self.vb_terms_bpd(
          model_fn=model_fn, x_start=x_start, t=t,
          x_t=self.q_sample(
              x_start=x_start, t=t,
              noise=jax.random.uniform(cur_rng, noise_shape)))
      del cur_rng
      assert vb.shape == (batch_size,)
      return vb

    vbterms_tb = lax.map(
        map_fn, (jnp.arange(self.num_timesteps),
                 jax.random.split(rng, self.num_timesteps)))
    vbterms_bt = vbterms_tb.T
    assert vbterms_bt.shape == (batch_size, self.num_timesteps)

    prior_b = self.prior_bpd(x_start=x_start)
    total_b = vbterms_tb.sum(axis=0) + prior_b
    assert prior_b.shape == total_b.shape == (batch_size,)

    return {
        'total': total_b,
        'vbterms': vbterms_bt,
        'prior': prior_b,
    }
