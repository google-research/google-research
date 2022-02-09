"""Gradient estimators.
"""
import pdb
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=('unroll_fn', 'T', 'K', 'N', 'sigma'))
def es_grad(key, key_for_data, unroll_fn, theta, state, T, K, N, sigma):
  """Computes the vanilla ES gradient estimate.

  Args:
    key: A JAX PRNGKey used for sampling random perturbations.
    key_for_data: A JAX PRNGKey passed to the unroll function for data sampling.
    unroll_fn: The unroll function for the inner problem.
    theta: The mean value of theta about which we sample perturbed thetas.
    state: A NamedTuple containing the inner problem state information.
    T: The total horizon length for a full sequence/inner problem
    K: The truncation length for a partial unroll.
    N: The number of particles/samples used to compute a Monte-Carlo estimate
       of the expectation defining the ES gradient estimate.
    sigma: The variance of perturbations.

  Returns:
    An estimate of the vanilla ES gradient formed by using N//2 antithetic
    noise pairs and unrolling the dynamical system for K steps with each
    perturbed value of theta.
  """
  pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
  neg_pert = -pos_pert
  perts = jnp.concatenate([pos_pert, neg_pert])
  obj, _ = jax.vmap(unroll_fn, in_axes=(None,0,None,None,None))(
    key_for_data, theta + perts, state, T, K
  )

  weighted_obj = obj.reshape(-1,1) * perts
  grad_estimate = jnp.sum(weighted_obj, axis=0) / (N * sigma**2)
  return grad_estimate


@partial(jax.jit, static_argnames=('unroll_fn', 'T', 'K', 'N', 'sigma'))
def pes_grad(key, key_for_data, unroll_fn, theta, state, T, K, N, sigma):
  """Computes the Persistent Evolution Strategies gradient estimate.

  Args:
    key: A JAX PRNGKey used for sampling random perturbations.
    key_for_data: A JAX PRNGKey passed to the unroll function for data sampling.
    unroll_fn: The unroll function for the inner problem.
    theta: The mean value of theta about which we sample perturbed thetas.
    state: A NamedTuple containing the inner problem state information.
    T: The total horizon length for a full sequence/inner problem
    K: The truncation length for a partial unroll.
    N: The number of particles/samples used to compute a Monte-Carlo estimate
       of the expectation defining the ES gradient estimate.
    sigma: The variance of perturbations.

  Returns:
    An estimate of the PES gradient formed by using N//2 antithetic noise
    pairs and unrolling the dynamical system for K steps with each perturbed
    value of theta.
  """
  pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
  neg_pert = -pos_pert
  perts = jnp.concatenate([pos_pert, neg_pert])
  obj, state_new = jax.vmap(unroll_fn, in_axes=(None,0,0,None,None))(
    key_for_data, theta + perts, state, T, K
  )
  state_new = state_new._replace(
    pert_accums=state_new.pert_accums + perts
  )

  weighted_obj = obj.reshape(-1, 1) * state_new.pert_accums
  grad_estimate = jnp.sum(weighted_obj, axis=0) / (N * sigma**2)
  return grad_estimate, state_new


@partial(jax.jit, static_argnames=('unroll_fn', 'T', 'K', 'N', 'sigma'))
def pes_grad_telescoping(key, key_for_data, unroll_fn, theta, state,
                         T, K, N, sigma):
  """Computes the Persistent Evolution Strategies gradient estimate with
  telescoping sums.

  Args:
    key: A JAX PRNGKey used for sampling random perturbations.
    key_for_data: A JAX PRNGKey passed to the unroll function for data sampling.
    unroll_fn: The unroll function for the inner problem.
    theta: The mean value of theta about which we sample perturbed thetas.
    state: A NamedTuple containing the inner problem state information.
    T: The total horizon length for a full sequence/inner problem
    K: The truncation length for a partial unroll.
    N: The number of particles/samples used to compute a Monte-Carlo estimate
       of the expectation defining the ES gradient estimate.
    sigma: The variance of perturbations.

  Returns:
    An estimate of the PES gradient formed by using N//2 antithetic noise
    pairs and unrolling the dynamical system for K steps with each perturbed
    value of theta.
  """
  pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
  neg_pert = -pos_pert
  perts = jnp.concatenate([pos_pert, neg_pert])

  prev_obj = state.prev_obj
  obj, state_new = jax.vmap(unroll_fn, in_axes=(None,0,0,None,None))(
    key_for_data, theta + perts, state, T, K
  )

  result = obj + prev_obj
  grad_estimate = jnp.sum(
      result.reshape(-1, 1) * (state_new.pert_accums + perts) -
      (prev_obj.reshape(-1, 1) * state_new.pert_accums),
      axis=0) / (N * sigma**2)

  state_new = state_new._replace(
    pert_accums=state_new.pert_accums + perts
  )
  return grad_estimate, state_new


@partial(jax.jit, static_argnames=('unroll_fn', 'T', 'K', 'N', 'sigma'))
def pes_grad_analytic(key, key_for_data, unroll_fn, theta, state, state_mean,
                      T, K, N, sigma):
  """Computes the PES+Analytic gradient estimate, which incorporates the
  analytic gradient from the most recent unroll.

  Args:
    key: A JAX PRNGKey used for sampling random perturbations.
    key_for_data: A JAX PRNGKey passed to the unroll function for data sampling.
    unroll_fn: The unroll function for the inner problem.
    theta: The mean value of theta about which we sample perturbed thetas.
    state: A NamedTuple containing the inner problem state information.
    state_mean: An extra (single) state that is evolved using the mean outer
                parameters theta (without perturbations).
    T: The total horizon length for a full sequence/inner problem
    K: The truncation length for a partial unroll.
    N: The number of particles/samples used to compute a Monte-Carlo estimate
       of the expectation defining the ES gradient estimate.
    sigma: The variance of perturbations.

  Returns:
    An estimate of the PES+Analytic gradient formed by using N//2 antithetic
    noise pairs and unrolling the dynamical system for K steps with each
    perturbed value of theta.
  """
  pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
  neg_pert = -pos_pert
  perts = jnp.concatenate([pos_pert, neg_pert])

  losses, state_new = jax.vmap(unroll_fn, in_axes=(None,0,0,None,None))(
    key_for_data, theta + perts, state, T, K
  )

  analytic_gradient, _ = jax.jit(
    jax.grad(unroll_fn, argnums=1, has_aux=True),
    static_argnames=('T', 'K')
  )(key_for_data, theta, state_mean, T, K)
  flat_analytic_grad = analytic_gradient
  things = losses - jnp.dot(perts, flat_analytic_grad)
  weighted_objective = things.reshape(-1, 1) * state_new.pert_accums
  grad_estimate = jnp.sum(weighted_objective, axis=0) / (N * sigma**2)
  state_new = state_new._replace(
    pert_accums=state_new.pert_accums + perts
  )
  grad_estimate = grad_estimate + flat_analytic_grad
  return grad_estimate, state_new


class ESParticleChunk:
  def __init__(self, key, K, T, sigma, N, init_state_fn, unroll_fn,
               initial_reset_t=None, telescoping=False):
    super().__init__()
    self.key, self.key_unroll, self.key_eval = jax.random.split(key, num=3)
    self.unroll_fn = unroll_fn
    self.init_state_fn = init_state_fn
    self.T = T
    self.K = K
    self.N = N
    self.sigma = sigma
    self.telescoping = telescoping
    self.has_been_reset = False
    self.initial_reset_t = initial_reset_t if initial_reset_t else T
    self.reset()

  def reset(self):
    self.state = self.init_state_fn(self.key)

    if self.telescoping:
      self.key_eval, skey_eval = jax.random.split(self.key_eval)

  def grad_estimate(self, theta, update_state=True):
    key, skey = jax.random.split(self.key)
    key_unroll, skey_unroll = jax.random.split(self.key_unroll)

    if self.telescoping:
      key_eval, skey_eval = self.key_eval, self.key_eval
    else:
      key_eval, skey_eval = jax.random.split(self.key_eval)

    skey_for_data = (skey_unroll, skey_eval)
    gradient = es_grad(
        skey, skey_for_data, self.unroll_fn, theta, self.state,
        self.T, self.K, N=self.N, sigma=self.sigma
    )

    if update_state:
      _, self.state = self.unroll_fn(
        skey_for_data, theta, self.state, self.T, self.K
      )

      self.key = key
      self.key_eval = key_eval
      self.key_unroll = key_unroll

      if (self.state.t >= self.T) or \
         (not self.has_been_reset and (self.state.t >= self.initial_reset_t)):
        self.has_been_reset = True
        self.reset()

    return gradient


class PESParticleChunk:
  def __init__(self, key, init_state_fn, unroll_fn, theta_shape,
               K, T, sigma, N, initial_reset_t=None, telescoping=False):
    super().__init__()
    self.key, self.key_unroll, self.key_eval = jax.random.split(key, num=3)
    self.T = T
    self.K = K
    self.N = N
    self.sigma = sigma
    self.theta_shape = theta_shape
    self.init_state_fn = init_state_fn
    self.unroll_fn = unroll_fn
    self.telescoping = telescoping
    self.has_been_reset = False
    self.initial_reset_t = initial_reset_t if initial_reset_t else T
    self.reset()

  def reset(self):
    self.state = jax.vmap(self.init_state_fn)(
      jnp.array([self.key] * self.N)
    )
    self.state = self.state._replace(
      pert_accums=jnp.zeros((self.N, self.theta_shape[0]))
    )

    if self.telescoping:
      self.key_eval, skey_eval = jax.random.split(self.key_eval)

  def grad_estimate(self, theta, update_state=True):
    key, skey = jax.random.split(self.key)
    key_unroll, skey_unroll = jax.random.split(self.key_unroll)

    if self.telescoping:
      pes_grad_fn = pes_grad_telescoping
      key_eval, skey_eval = self.key_eval, self.key_eval
    else:
      pes_grad_fn = pes_grad
      key_eval, skey_eval = jax.random.split(self.key_eval)

    skey_for_data = (skey_unroll, skey_eval)
    gradient, state_updated = pes_grad_fn(
        skey, skey_for_data, self.unroll_fn, theta, self.state,
        self.T, self.K, N=self.N, sigma=self.sigma
    )

    if update_state:
      self.key = key
      self.key_eval = key_eval
      self.key_unroll = key_unroll
      self.state = state_updated
      if (self.state.t[0] >= self.T) or \
         (not self.has_been_reset and (self.state.t[0] >= self.initial_reset_t)):
        self.has_been_reset = True
        self.reset()

    return gradient


class PESAParticleChunk:
  def __init__(self, key, init_state_fn, unroll_fn, theta_shape,
               K, T, sigma, N, initial_reset_t=None, telescoping=False):
    super().__init__()
    self.key, self.key_for_data = jax.random.split(key)
    self.T = T
    self.K = K
    self.N = N
    self.sigma = sigma
    self.theta_shape = theta_shape
    self.init_state_fn = init_state_fn
    self.unroll_fn = unroll_fn
    self.telescoping = telescoping
    self.has_been_reset = False
    self.initial_reset_t = initial_reset_t if initial_reset_t else T
    self.reset()

  def reset(self):
    self.state = jax.vmap(self.init_state_fn)(
      jnp.array([self.key] * self.N)
    )
    self.state = self.state._replace(
      pert_accums=jnp.zeros((self.N, self.theta_shape[0]))
    )

    # Initialize an additional state that will be evolved using the mean thetas
    self.state_mean = self.init_state_fn(self.key)

    if self.telescoping:
      self.key_for_data, skey_for_data = jax.random.split(self.key_for_data)
      self.state = self.state._replace(prev_obj=jnp.zeros((self.N,)))
      self.state_mean = self.state_mean._replace(prev_obj=jnp.array(0.0))

  def grad_estimate(self, theta, update_state=True):
    key, skey = jax.random.split(self.key)

    if self.telescoping:
      key_for_data, skey_for_data = self.key_for_data, self.key_for_data
    else:
      key_for_data, skey_for_data = jax.random.split(self.key_for_data)

    gradient, state_updated = pes_grad_analytic(
        skey, skey_for_data, self.unroll_fn, theta, self.state, self.state_mean,
        self.T, self.K, N=self.N, sigma=self.sigma
    )

    if update_state:
      # Update state_mean using the mean thetas
      _, self.state_mean = self.unroll_fn(
          skey_for_data, theta, self.state_mean, self.T, self.K
      )

      self.key = key
      self.key_for_data = key_for_data
      self.state = state_updated

      if (self.state.t[0] >= self.T) or \
         (not self.has_been_reset and (self.state.t[0] >= self.initial_reset_t)):
        self.has_been_reset = True
        self.reset()

    return gradient


class TBPTTParticle:
  def __init__(self, key, init_state_fn, unroll_fn, K, T, initial_reset_t=None):
    super().__init__()

    self.key = key
    self.T = T
    self.K = K
    self.unroll_fn = unroll_fn
    self.init_state_fn = init_state_fn

    self.has_been_reset = False
    self.initial_reset_t = initial_reset_t if initial_reset_t else T

    self.reset()

    self.unroll_grad = jax.jit(
        jax.grad(unroll_fn, argnums=1, has_aux=True),
        static_argnames=('T', 'K')
    )

  def reset(self):
    self.state = self.init_state_fn(self.key)

  def grad_estimate(self, theta, update_state=True):
    key, skey = jax.random.split(self.key)
    gradient, _ = self.unroll_grad(
        skey, theta, self.state, self.T, self.K
    )

    if update_state:
      self.key = key
      _, self.state = self.unroll_fn(
          skey, theta, self.state, self.T, self.K
      )

      if (self.state.t >= self.T) or \
         (not self.has_been_reset and (self.state.t >= self.initial_reset_t)):
        self.has_been_reset = True
        self.reset()

    return gradient


class RTRLParticle:
  def __init__(self, key, init_state_fn, unroll_fn, K, T, initial_reset_t=None):
    super().__init__()

    self.key = key
    self.T = T
    self.K = K
    self.unroll_fn = unroll_fn
    self.init_state_fn = init_state_fn

    self.has_been_reset = False
    self.initial_reset_t = initial_reset_t if initial_reset_t else T

    def f(theta, state):
      loss, updated_state = self.unroll_fn(None, theta, state, self.T, 1)
      return updated_state.inner_state

    def L(theta, state):
      loss, updated_state = self.unroll_fn(None, theta, state, self.T, 1)
      return loss

    self.compute_d_state_new_d_theta_direct = jax.jit(jax.jacrev(f, argnums=0))
    self.compute_d_state_new_d_state_old = jax.jit(jax.jacrev(f, argnums=1))
    self.compute_dL_dstate_old = jax.jit(jax.grad(L, argnums=1))
    self.compute_dL_dtheta_direct = jax.jit(jax.grad(L, argnums=0))
    self.compute_loss_grad_bptt = jax.jit(jax.grad(L, argnums=0))

    self.reset()

  def reset(self):
    self.dstate_dtheta = None
    self.state = self.init_state_fn(self.key)

  def rtrl_grad(self, theta, state, dstate_dtheta):
    total_theta_grad = 0
    total_loss = 0.0

    if dstate_dtheta is None:
      dstate_dtheta = jnp.zeros((len(state.inner_state), len(theta)))

    state_old = state
    loss, state_new = self.unroll_fn(None, theta, state_old, self.T, 1)
    total_loss += loss

    dl_dstate_old = self.compute_dL_dstate_old(theta, state_old)
    dl_dtheta_direct = self.compute_dL_dtheta_direct(theta, state_old)

    d_state_new_d_state_old = self.compute_d_state_new_d_state_old(
        theta, state_old
    )
    d_state_new_d_theta_direct = self.compute_d_state_new_d_theta_direct(
        theta, state_old
    )

    dl_dstate_old = dl_dstate_old.inner_state
    d_state_new_d_state_old = d_state_new_d_state_old.inner_state

    theta_grad = jnp.dot(dl_dstate_old.reshape(1, -1),
                         dstate_dtheta).reshape(-1) + dl_dtheta_direct
    total_theta_grad += theta_grad
    dstate_dtheta = jnp.dot(d_state_new_d_state_old, dstate_dtheta) \
                    + d_state_new_d_theta_direct

    return (total_loss, state_new, dstate_dtheta), total_theta_grad

  def grad_estimate(self, theta, update_state=True):
    (loss, state_updated, dstate_dtheta_updated), gradient = self.rtrl_grad(
        theta, self.state, self.dstate_dtheta
    )

    if update_state:
      self.state = state_updated
      self.dstate_dtheta = dstate_dtheta_updated

      if (self.state.t >= self.T) or \
         (not self.has_been_reset and (self.state.t >= self.initial_reset_t)):
        self.has_been_reset = True
        self.reset()

    return gradient


class UOROParticle:
  def __init__(self, key, init_state_fn, unroll_fn, K, T, initial_reset_t=None):
    super().__init__()

    self.key = key
    self.T = T
    self.K = K
    self.unroll_fn = unroll_fn
    self.init_state_fn = init_state_fn

    self.has_been_reset = False
    self.initial_reset_t = initial_reset_t if initial_reset_t else T

    def f(theta, state):
      loss, updated_state = self.unroll_fn(None, theta, state, self.T, 1)
      return updated_state.inner_state

    def L(theta, state):
      loss, updated_state = self.unroll_fn(None, theta, state, self.T, 1)
      return loss

    self.f = f
    self.L = L
    self.compute_d_state_new_d_theta_direct = jax.jit(jax.jacrev(f, argnums=0))
    self.compute_d_state_new_d_state_old = jax.jit(jax.jacrev(f, argnums=1))
    self.compute_dL_dstate_old = jax.jit(jax.grad(L, argnums=1))
    self.compute_dL_dtheta_direct = jax.jit(jax.grad(L, argnums=0))
    self.compute_loss_grad_bptt = jax.jit(jax.grad(L, argnums=0))

    self.reset()

  def reset(self):
    self.s_tilde = None
    self.theta_tilde = None
    self.state = self.init_state_fn(self.key)

  def uoro_grad(self, key, theta, state, s_tilde=None, theta_tilde=None):
    epsilon_perturbation = 1e-7
    epsilon_stability = 1e-7

    total_theta_grad = 0
    total_loss = 0.0

    if s_tilde is None:
      s_tilde = jnp.zeros(state.inner_state.shape)

    if theta_tilde is None:
      theta_tilde = jnp.zeros(theta.shape)

    state_old = state
    # TODO: How do we handle key here? Do we want to split again?
    loss, state_new = self.unroll_fn(key, theta, state_old, self.T, 1)
    total_loss += loss

    dl_dstate_old = self.compute_dL_dstate_old(theta, state_old)
    dl_dtheta_direct = self.compute_dL_dtheta_direct(theta, state_old)

    dl_dstate_old = dl_dstate_old.inner_state

    indirect_grad = (dl_dstate_old * s_tilde).sum() * theta_tilde
    pseudograds = indirect_grad + dl_dtheta_direct

    state_old_perturbed = state_old._replace(
        inner_state=state_old.inner_state + s_tilde * epsilon_perturbation
    )
    state_new_perturbed = self.f(theta, state_old_perturbed)

    state_deriv_in_direction_s_tilde = (
        (state_new_perturbed - state_new.inner_state) / epsilon_perturbation
    )

    nus = jnp.round(jax.random.uniform(key, state_old.inner_state.shape)) * 2 - 1

    custom_f = lambda param_vector: self.f(param_vector, state_old)
    primals, f_vjp = jax.vjp(custom_f, theta)
    direct_theta_tilde_contribution, = f_vjp(nus)

    rho_0 = jnp.sqrt((jnp.linalg.norm(theta_tilde) + epsilon_stability)
                      / (jnp.linalg.norm(state_deriv_in_direction_s_tilde)
                      + epsilon_stability))
    rho_1 = jnp.sqrt(
        (jnp.linalg.norm(direct_theta_tilde_contribution) + epsilon_stability)
        / (jnp.linalg.norm(nus) + epsilon_stability)
    )

    theta_grad = pseudograds
    total_theta_grad += theta_grad

    s_tilde = rho_0 * state_deriv_in_direction_s_tilde + rho_1 * nus
    theta_tilde = theta_tilde / rho_0 + direct_theta_tilde_contribution / rho_1

    return (total_loss, state_new, s_tilde, theta_tilde), total_theta_grad

  def grad_estimate(self, theta, update_state=True):
    key, skey = jax.random.split(self.key)
    (loss, state_updated, s_tilde, theta_tilde), gradient = self.uoro_grad(
      skey, theta, self.state, self.s_tilde, self.theta_tilde
    )

    if update_state:
      self.key = key
      self.state = state_updated
      self.s_tilde = s_tilde
      self.theta_tilde = theta_tilde

      if (self.state.t >= self.T) or \
         (not self.has_been_reset and (self.state.t >= self.initial_reset_t)):
        self.has_been_reset = True
        self.reset()

    return gradient


class MultiParticleEstimator:
  def __init__(self, key, theta_shape, n_chunks,
               n_particles_per_chunk, K, T, sigma,
               init_state_fn, unroll_fn, method='lockstep',
               estimator_type='es', telescoping=False):
    super().__init__()

    if T is None:
      # Set T to indicate that the inner problem never ends
      T = float('inf')

    self.init_state_fn = init_state_fn
    self.unroll_fn = unroll_fn
    self.n_chunks = n_chunks
    self.telescoping = telescoping

    keys = jax.random.split(key, n_chunks)
    self.particle_chunks = []
    for i in range(n_chunks):
      if method == 'breakstep':
        initial_reset_t = (T // n_chunks) + i * (T // n_chunks)
        print(initial_reset_t)
      else:
        initial_reset_t = None

      if estimator_type == 'es':
        particle_chunk = ESParticleChunk(
          key=keys[i], K=K, T=T, sigma=sigma,
          N=n_particles_per_chunk, initial_reset_t=initial_reset_t,
          init_state_fn=init_state_fn, unroll_fn=unroll_fn,
          telescoping=telescoping
        )
      elif estimator_type == 'pes':
        particle_chunk = PESParticleChunk(
          key=keys[i], theta_shape=theta_shape, K=K, T=T, sigma=sigma,
          N=n_particles_per_chunk, initial_reset_t=initial_reset_t,
          init_state_fn=init_state_fn, unroll_fn=unroll_fn,
          telescoping=telescoping
        )
      elif estimator_type == 'pes-a':
        particle_chunk = PESAParticleChunk(
          key=keys[i], theta_shape=theta_shape, K=K, T=T, sigma=sigma,
          N=n_particles_per_chunk, initial_reset_t=initial_reset_t,
          init_state_fn=init_state_fn, unroll_fn=unroll_fn,
          telescoping=telescoping
        )
      elif estimator_type == 'tbptt':
        particle_chunk = TBPTTParticle(
          key=keys[i], K=K, T=T, initial_reset_t=initial_reset_t,
          init_state_fn=init_state_fn, unroll_fn=unroll_fn
        )
      elif estimator_type == 'rtrl':
        particle_chunk = RTRLParticle(
          key=keys[i], K=K, T=T, initial_reset_t=initial_reset_t,
          init_state_fn=init_state_fn, unroll_fn=unroll_fn
        )
      elif estimator_type == 'uoro':
        particle_chunk = UOROParticle(
          key=keys[i], K=K, T=T, initial_reset_t=initial_reset_t,
          init_state_fn=init_state_fn, unroll_fn=unroll_fn
        )
      self.particle_chunks.append(particle_chunk)

  def grad_estimate(self, theta, update_state=True):
    total_gradient = 0.0
    for particle_chunk in self.particle_chunks:
      total_gradient += particle_chunk.grad_estimate(
          theta, update_state=update_state
      )
    return total_gradient / float(self.n_chunks)
