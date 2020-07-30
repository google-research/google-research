Munchausen Reinforcement Learning
===
Nino Vieillard, Olivier Pietquin, Matthieu Geist
---

This repository contains code related to the paper [*Munchausen Reinforcement
Learning*](https://arxiv.org/abs/2007.14430) (M-RL).

M-RL is based on the [Dopamine](https://github.com/google/dopamine) library
([Castro et al., *Dopamine: A Research Framework for Deep Reinforcement Learning*, 2018](http://arxiv.org/abs/1812.06110)).


## General description
The paper introduces two new agents: Munchausen DQN (M-DQN) and munchausen IQN
(M-IQN). These agents are implemented on top of Dopamine agents in respectively
agents/m_dqn.py and agents/m_iqn.py.

In addition, we provide an implementation of Advantage Learning, used in our
ablation study, in agents/al_dqn.py.

## Usage
To train an agent, use the `train.py` script. For example:
```
# From google_research/:

python -m munchausen_rl.train --env atari --game Pong --agent_type m_dqn --workdir /tmp/munchausen --gin_file munchausen_rl/configs/atari.gin
```
will train M-DQN on the Pong environment.

For detailed usage, run
```
python -m munchausen_rl.train --help
```

## Dependencies
M-RL is compatible with python >= 3.7.
You can install all necessary requirements by running:
```
pip install -r munchausen_rl/requirements.txt`.
```

__NB:__ this implementation of M-RL is implemented in tensoflow v1, and does not
support eager mode.
