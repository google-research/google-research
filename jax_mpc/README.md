# JAX Particles

This repo contains model predictive controllers written in JAX. JAX is
particularly suited for monte-carlo style MPC, as rollouts can be
efficiently parallelized using `jax.vmap()`.

### Installation

```shell
$ pip install .
```

### Demo

```python
from mpc.mppi import MPPI

# let "env" be an environment object with
# members:
#   a_shape: describing the shape of actions
# methods:
#   step(s, a): returns a new state
#   reward(s): returns an environment reward for state s
# see method descriptions of class MPPI for specifics.

# let "s" be an environment state of unspecified shape compatible with env

# instantiate the MPC object
mpc = MPPI(n_iterations=5, n_steps=16, n_samples=16, scan=True)

# initialize the state
mpc_state = mpc.init_state(env.a_shape)

# loop

# do work to identify a good action
mpc_state = mpc.update(mpc_state, env, s, rng)

# get the recommended action
a = mpc.get_action(mpc_state, env.a_shape)

# take the action in the environment
s = env.step(s,a)
```
