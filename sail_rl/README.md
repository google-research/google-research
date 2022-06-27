Self-Imitation Advantage Learning
===
Johan Ferret, Olivier Pietquin, Matthieu Geist
---

This repository contains code related to the paper [*Self-Imitation Advantage Learning*](https://arxiv.org/abs/2012.11989) (SAIL) which is to appear at AAMAS 2021.

This implementation of SAIL is based on the [Dopamine](https://github.com/google/dopamine) library
([Castro et al., *Dopamine: A Research Framework for Deep Reinforcement Learning*, 2018](http://arxiv.org/abs/1812.06110)).


## General description
The paper introduces two new agents: SAIL-DQN and SAIL-IQN.
These agents are implemented on top of Dopamine agents and are
respectively located in `agents/sail_dqn.py` and `agents/sail_iqn.py`.

In addition, we provide an implementation of Advantage Learning, used in our
ablation study, in `agents/al_dqn.py` and `agents/al_iqn.py`.

## Usage
To train an agent, use the `train.py` script. For example:
```
# From google_research/:

python -m sail_rl.train --env atari --game Pong --agent_type sail_dqn --workdir /tmp/sail --gin_file sail_rl/configs/atari.gin
```
will train SAIL-DQN on the Pong environment.

For detailed usage, run
```
python -m sail_rl.train --help
```

Note that for improved performance of the AL-DQN and SAIL-DQN agents, Adam should be used in place of RMSProp. This can be changed in the Gin config files. We keep RMSProp in order to stay as close as possible to the paper's experimental setup.

## Dependencies
SAIL is compatible with python >= 3.7.
You can install all necessary requirements by running:
```
pip install -r sail_rl/requirements.txt`.
```

Atari ROMs should be downloaded as well.

__NB:__ this implementation of SAIL uses tensorflow v1, and does not
support Eager mode.
