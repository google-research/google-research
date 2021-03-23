# Joint attention extension to multi-agent tf-agents

This repo implements an extension to the multi-agent tf-agents training code 
for the paper "Joint Attention for Multi-Agent Coordination and Social Learning"
(link to be added).
It modifies the agents' architecture to produce attention maps over the image
input, and compares these attention maps to compute joint attention bonuses.
Incentivizing joint attention allows agents to learn to coordinate with each
other more efficiently by mutually attending to the same parts of the
environment.


## Training

To train multiple PPO agents in the randomized `Meetup` environment
(implemented in `gym_multigrid/envs/meetup.py`, use:

```
python -m joint_attention_train_eval.py --debug --root_dir=/tmp/meetup/ \
--env_name='MultiGrid-Meetup-Empty-12x12-v0'
```

