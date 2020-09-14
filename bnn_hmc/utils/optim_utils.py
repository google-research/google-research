import jax
import optax
import jax.numpy as jnp
import numpy as onp


def make_sgd_optimizer(lr_schedule, momentum_decay):
  """Make SGD optimizer with momentum."""
  # Maximize log-prob instead of minimizing loss
  return optax.chain(optax.trace(decay=momentum_decay, nesterov=False),
                     optax.scale_by_schedule(lr_schedule))


def make_adam_optimizer(lr_schedule, b1=0.9, b2=0.999, eps=1e-8):
  """Make Adam optimizer."""
  # Maximize log-prob instead of minimizing loss
  return optax.chain(optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
                     optax.scale_by_schedule(lr_schedule))


def make_cosine_lr_schedule(init_lr, total_steps):
  """Cosine LR schedule."""
  def schedule(step):
    t = step / total_steps
    return 0.5 * init_lr * (1 + jnp.cos(t * onp.pi))
  return schedule


def make_constant_lr_schedule_with_cosine_burnin(
    init_lr, final_lr, burnin_steps
):
  """Cosine LR schedule with burn-in for SG-MCMC."""
  def schedule(step):
    t = jnp.minimum(step / burnin_steps, 1.)
    coef = (1 + jnp.cos(t * onp.pi)) * 0.5
    return coef * init_lr + (1 - coef) * final_lr
  return schedule


def make_cyclical_cosine_lr_schedule_with_const_burnin(
    init_lr, burnin_steps, cycle_length
):
  def schedule(step):
    t = jnp.maximum(step - burnin_steps - 1, 0.)
    t = (t % cycle_length) / cycle_length
    return 0.5 * init_lr * (1 + jnp.cos(t * onp.pi))
  return schedule