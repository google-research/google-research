# Multi-agent reinforcement learning implemented in tf-agents

This repo implements multi-agent RL agent training code on top of tf-agents. 
There is a centralized meta-agent, that contains and initializes several other
PPO agents. The meta-agent receives an aggregated dictionary observation from
the environment, containing the observations of all agents. It then splits these 
up and passes them to the sub-agents, which return their actions. The actions
are aggregated and returned to the environment, which computes a vector reward
containing the individual reward of all agents. 

## Meta-agent implementation

The meta-agent is implemented in `multiagent_ppo.py` and
`multiagent_ppo_policy.py`. It uses multi-agent specs, and expects to receive
multiple observations and rewards, and output multiple actions.

The architecture for the individual agents is implemented in
`multigrid_networks.py`. This repo also implements modifications to the
tf-agents environments and metrics to make them compatible with multi-agent
training.

## Training

To train multiple PPO agents in the randomized `Cluttered` environment
(implemented in `gym_multigrid/envs/cluttered.py`, use:

```
python -m multiagent_train_eval.py --debug --root_dir=/tmp/cluttered/ \
--env_name='MultiGrid-Cluttered-v0'
```

To train a single agent, use:
```
python -m single_agent_train_eval.py --debug --root_dir=/tmp/single_cluttered/ \
--env_name='MultiGrid-Cluttered-Single-6x6-v0' 
```
