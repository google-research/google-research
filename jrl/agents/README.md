# Agents

For instructions about running each agent please refer to the `README.md` file
in the corresponding agent's directory.

Here we include an overview of how
agents are setup in this codebase.
The MSG implementation demonstrates the typical manner in which new algorithms
(called “agents” in the codebase) are added to the jrl codebase.

## RL Component: e.g. `jrl/agents/msg/__init__.py`
An RL component defines how to create a Builder (in the Acme sense), make all
the necessary networks, a behavior policy (in case doing online rollouts), and
an eval behavior policy. When creating a new agent/algorithm, it’s RL component
must be added to `jrl/agents/__init__.py`.

## Builder: e.g. `jrl/agents/msg/builder.py`
Builder is an Acme builder which for the sake of offline RL defines how to get
the learner, and how to create the actor. You can see that some functions return
None or [] because they are needed only for online RL.

## Config: e.g. `jrl/agents/msg/config.py`
In the jrl codebase, algorithms are configured using gin. Specifically, each
algorithm contains a config dataclass object which is gin configurable. This
config is then piped to various components such as the builder, learner, etc.

## Learner: e.g. `jrl/agents/msg/learning.py`
The core of the implementation of offline RL algorithms is the learning.py file.

## Networks: e.g. `jrl/agents/msg/networks.py`
If you remember from above, the RLComponent has a make_networks method. I
thought it would be cleanest if this method called networks.make_networks,
because it can also pass in parameters from the gin configured Config dataclass
object. networks.make_networks returns a dataclass object containing all the
networks that might be needed for the learner, actor, eval policy, etc. An
important point to note is that since this is a Jax implementation, by networks,
I am actually referring to pure Jax functions (or containers of pure Jax
functions e.g. networks_lib.FeedForwardNetwork). By pure Jax functions I am
referring for example to the `apply_fn` that you obtain after performing
`haiku.transform`. As you can see in the MSG
case, I like to use haiku to define my neural architectures, then I call
transform to get pure functions. You can use whichever library you like most,
e.g. Flax, Objax, etc.
