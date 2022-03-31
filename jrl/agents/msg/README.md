# MSG

Reference implementation for MSG with deep ensembles, as well as efficient
alternatives such as MSG with MIMO ensembles, and MSG with Multi-Head ensembles.

Note: `msg/learning.py` may appear dauntingly large. However, note that
implementation in this codebase contains a variety of MSG variations (some which
did not make it into the paper), as well orthogonal research ideas. For the
interested reader, we recommend starting at `def _full_update_step` and
following the function calls for the version of MSG that you are interested in.

Note: For advice on how to implement your own variation of MSG, please refer
to the included section below.

Please remember that some flags are "global", meaning that they are defined in
`runner_flags.py`, while algorithm specific parameters are configured using
gin configs.

All agents have a config parameter named `num_sgd_steps_per_step`, which
determines how many training steps we perform per call to the learner's step
function. Setting this to a larger number allows Jax to perform optimizations
that make training faster. Keep in mind that you should set the `batch_size`
parameter to `(num_sgd_steps_per_step) x (per training step batch size that you want)`.
Also, you should set `num_steps` to
`(total number of training steps you want) / num_sgd_steps_per_step`.
`msg.config.MSGConfig.num_bc_iters` and `msg.config.MSGConfig.pretrain_iters`
are set in terms of "true" number of steps, i.e. no need to account for
`num_sgd_steps_per_step`.
For easier local debugging, you can set:
```
--num_sgd_steps_per_step 1 \
--batch_size 64 \
--num_steps 1000 \
--episodes_per_eval 10 \
--gin_bindings='msg.config.MSGConfig.num_sgd_steps_per_step=1'
```

Note: Only for `halfcheetah, hopper, walker` experiments we set
`msg.config.MSGConfig.use_double_q=True`. It was not a noticeable difference
but we did not go back to rerun full experiments with this param set to True.

Note: Using SASS (by setting `msg.config.MSGConfig.use_sass=True`) can help
a decent bit. For example in D4RL antmaze it provides a 10-15% percent boost
on our paper results for MSG with deep ensembles. When `msg.config.MSGConfig.use_sass=True` (or `msg.config.MSGConfig.num_q_repr_pretrain_iters > 0`), the dataloader
must also `return next_actions`.

Note: For additional parameters that can be set please refer to `msg/config.py`

## Advice About Implementing Alternative Forms of MSG
In this codebase we implemented MSG with deep ensembles differently from MSG
with MIMO or Multi-Head ensembles. The reason for implementing deep ensembles
differently was to enable multi-GPU or TPU training (since Jax treats each core
of a TPU as a different device). While the implementation is correct and tested,
all of the experiments in our work were done in a single-GPU setting, and thus
we implemented MIMO and Multi-Head ensembles for single-device settings. As a
result, if you are interested in implementing your own variation of MSG, we
would recommend following the implementation for MIMO, Multi-Head, or Tree Deep
Ensembles.


## Running MSG with deep ensembles
```
python3 -m jrl.localized.runner \
--pdb_post_mortem \
--debug_nans=False \
--create_saved_model_actor=False \
--num_steps 11000 \
--eval_every_steps 500 \
--episodes_per_eval 100 \
--batch_size 51200 \
--root_dir '/tmp/test_msg_deep_ensembles' \
--seed 42 \
--algorithm 'msg' \
--task_class 'd4rl' \
--task_name 'antmaze-large-diverse-v0' \
--gin_bindings='msg.config.MSGConfig.num_sgd_steps_per_step=200' \
--gin_bindings='msg.config.MSGConfig.ensemble_size=64' \
--gin_bindings='msg.config.MSGConfig.ensemble_method="deep_ensembles"' \
--gin_bindings='msg.config.MSGConfig.td_target_method="independent"' \
--gin_bindings='msg.config.MSGConfig.beta=-8' \
--gin_bindings='msg.config.MSGConfig.behavior_regularization_alpha=0.1' \
--gin_bindings='msg.config.MSGConfig.behavior_regularization_type="v1"' \
--gin_bindings='msg.config.MSGConfig.num_cql_actions=1' \
--gin_bindings='msg.config.MSGConfig.use_random_weighting_in_critic_loss=True' \
--gin_bindings='msg.config.MSGConfig.num_bc_iters=50000' \
--gin_bindings='msg.config.MSGConfig.num_q_repr_pretrain_iters=0' \
--gin_bindings='msg.config.MSGConfig.pretrain_temp=1' \
--gin_bindings='msg.config.MSGConfig.use_sass=False' \
--gin_bindings='msg.config.MSGConfig.q_lr=3e-4' \
--gin_bindings='msg.config.MSGConfig.policy_lr=3e-5' \
--gin_bindings='msg.config.MSGConfig.use_ema_target_critic_params=True' \
--gin_bindings='msg.config.MSGConfig.use_entropy_regularization=True' \
--gin_bindings='msg.config.MSGConfig.actor_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='msg.config.MSGConfig.critic_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='msg.config.MSGConfig.use_double_q=False' \
--gin_bindings='msg.config.MSGConfig.networks_init_type="glorot_also_dist"' \
--gin_bindings='msg.config.MSGConfig.critic_random_init=False' \
--gin_bindings='msg.config.MSGConfig.perform_sarsa_q_eval=False' \
--gin_bindings='msg.config.MSGConfig.eval_with_q_filter=False' \
--gin_bindings='msg.config.MSGConfig.num_eval_samples=32' \
--gin_bindings='msg.config.MSGConfig.mimo_using_adamw=False' \
--gin_bindings='msg.config.MSGConfig.mimo_using_obs_tile=False' \
--gin_bindings='msg.config.MSGConfig.mimo_using_act_tile=False'
```

## Running MSG with MIMO
```
python3 -m jrl.localized.runner \
--pdb_post_mortem \
--debug_nans=False \
--create_saved_model_actor=False \
--num_steps 11000 \
--eval_every_steps 500 \
--episodes_per_eval 100 \
--batch_size 51200 \
--root_dir '/tmp/test_msg_mimo' \
--seed 42 \
--algorithm 'msg' \
--task_class 'd4rl' \
--task_name 'antmaze-large-diverse-v0' \
--gin_bindings='msg.config.MSGConfig.num_sgd_steps_per_step=200' \
--gin_bindings='msg.config.MSGConfig.ensemble_size=64' \
--gin_bindings='msg.config.MSGConfig.ensemble_method="mimo"' \
--gin_bindings='msg.config.MSGConfig.td_target_method="independent"' \
--gin_bindings='msg.config.MSGConfig.beta=-8' \
--gin_bindings='msg.config.MSGConfig.behavior_regularization_alpha=0.1' \
--gin_bindings='msg.config.MSGConfig.behavior_regularization_type="v1"' \
--gin_bindings='msg.config.MSGConfig.num_cql_actions=1' \
--gin_bindings='msg.config.MSGConfig.use_random_weighting_in_critic_loss=True' \
--gin_bindings='msg.config.MSGConfig.num_bc_iters=50000' \
--gin_bindings='msg.config.MSGConfig.num_q_repr_pretrain_iters=0' \
--gin_bindings='msg.config.MSGConfig.pretrain_temp=1' \
--gin_bindings='msg.config.MSGConfig.use_sass=False' \
--gin_bindings='msg.config.MSGConfig.q_lr=3e-4' \
--gin_bindings='msg.config.MSGConfig.policy_lr=3e-5' \
--gin_bindings='msg.config.MSGConfig.use_ema_target_critic_params=True' \
--gin_bindings='msg.config.MSGConfig.use_entropy_regularization=True' \
--gin_bindings='msg.config.MSGConfig.actor_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='msg.config.MSGConfig.critic_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='msg.config.MSGConfig.use_double_q=False' \
--gin_bindings='msg.config.MSGConfig.networks_init_type="glorot_also_dist"' \
--gin_bindings='msg.config.MSGConfig.critic_random_init=False' \
--gin_bindings='msg.config.MSGConfig.perform_sarsa_q_eval=False' \
--gin_bindings='msg.config.MSGConfig.eval_with_q_filter=False' \
--gin_bindings='msg.config.MSGConfig.num_eval_samples=32' \
--gin_bindings='msg.config.MSGConfig.mimo_using_adamw=False' \
--gin_bindings='msg.config.MSGConfig.mimo_using_obs_tile=True' \
--gin_bindings='msg.config.MSGConfig.mimo_using_act_tile=True'
```

## Running MSG with Multi-Head ensembles
Multi-head is implemented as MIMO without the tiling of observations and
actions.

```
python3 -m jrl.localized.runner \
--pdb_post_mortem \
--debug_nans=False \
--create_saved_model_actor=False \
--num_steps 11000 \
--eval_every_steps 500 \
--episodes_per_eval 100 \
--batch_size 51200 \
--root_dir '/tmp/test_msg_multihead' \
--seed 42 \
--algorithm 'msg' \
--task_class 'd4rl' \
--task_name 'antmaze-large-diverse-v0' \
--gin_bindings='msg.config.MSGConfig.num_sgd_steps_per_step=200' \
--gin_bindings='msg.config.MSGConfig.ensemble_size=64' \
--gin_bindings='msg.config.MSGConfig.ensemble_method="mimo"' \
--gin_bindings='msg.config.MSGConfig.td_target_method="independent"' \
--gin_bindings='msg.config.MSGConfig.beta=-8' \
--gin_bindings='msg.config.MSGConfig.behavior_regularization_alpha=0.1' \
--gin_bindings='msg.config.MSGConfig.behavior_regularization_type="v1"' \
--gin_bindings='msg.config.MSGConfig.num_cql_actions=1' \
--gin_bindings='msg.config.MSGConfig.use_random_weighting_in_critic_loss=True' \
--gin_bindings='msg.config.MSGConfig.num_bc_iters=50000' \
--gin_bindings='msg.config.MSGConfig.num_q_repr_pretrain_iters=0' \
--gin_bindings='msg.config.MSGConfig.pretrain_temp=1' \
--gin_bindings='msg.config.MSGConfig.use_sass=False' \
--gin_bindings='msg.config.MSGConfig.q_lr=3e-4' \
--gin_bindings='msg.config.MSGConfig.policy_lr=3e-5' \
--gin_bindings='msg.config.MSGConfig.use_ema_target_critic_params=True' \
--gin_bindings='msg.config.MSGConfig.use_entropy_regularization=True' \
--gin_bindings='msg.config.MSGConfig.actor_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='msg.config.MSGConfig.critic_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='msg.config.MSGConfig.use_double_q=False' \
--gin_bindings='msg.config.MSGConfig.networks_init_type="glorot_also_dist"' \
--gin_bindings='msg.config.MSGConfig.critic_random_init=False' \
--gin_bindings='msg.config.MSGConfig.perform_sarsa_q_eval=False' \
--gin_bindings='msg.config.MSGConfig.eval_with_q_filter=False' \
--gin_bindings='msg.config.MSGConfig.num_eval_samples=32' \
--gin_bindings='msg.config.MSGConfig.mimo_using_adamw=False' \
--gin_bindings='msg.config.MSGConfig.mimo_using_obs_tile=False' \
--gin_bindings='msg.config.MSGConfig.mimo_using_act_tile=False'
```

## Running MSG with tree deep ensembles
```
python3 -m jrl.localized.runner \
--pdb_post_mortem \
--debug_nans=False \
--create_saved_model_actor=False \
--num_steps 11000 \
--eval_every_steps 500 \
--episodes_per_eval 100 \
--batch_size 51200 \
--root_dir '/tmp/test_msg_tree_deep_ensembles' \
--seed 42 \
--algorithm 'msg' \
--task_class 'd4rl' \
--task_name 'antmaze-large-diverse-v0' \
--gin_bindings='msg.config.MSGConfig.num_sgd_steps_per_step=200' \
--gin_bindings='msg.config.MSGConfig.ensemble_size=64' \
--gin_bindings='msg.config.MSGConfig.ensemble_method="tree_deep_ensembles"' \
--gin_bindings='msg.config.MSGConfig.td_target_method="independent"' \
--gin_bindings='msg.config.MSGConfig.beta=-8' \
--gin_bindings='msg.config.MSGConfig.behavior_regularization_alpha=0.1' \
--gin_bindings='msg.config.MSGConfig.behavior_regularization_type="v1"' \
--gin_bindings='msg.config.MSGConfig.num_cql_actions=1' \
--gin_bindings='msg.config.MSGConfig.use_random_weighting_in_critic_loss=True' \
--gin_bindings='msg.config.MSGConfig.num_bc_iters=50000' \
--gin_bindings='msg.config.MSGConfig.num_q_repr_pretrain_iters=0' \
--gin_bindings='msg.config.MSGConfig.pretrain_temp=1' \
--gin_bindings='msg.config.MSGConfig.use_sass=False' \
--gin_bindings='msg.config.MSGConfig.q_lr=3e-4' \
--gin_bindings='msg.config.MSGConfig.policy_lr=3e-5' \
--gin_bindings='msg.config.MSGConfig.use_ema_target_critic_params=True' \
--gin_bindings='msg.config.MSGConfig.use_entropy_regularization=True' \
--gin_bindings='msg.config.MSGConfig.actor_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='msg.config.MSGConfig.critic_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='msg.config.MSGConfig.use_double_q=False' \
--gin_bindings='msg.config.MSGConfig.networks_init_type="glorot_also_dist"' \
--gin_bindings='msg.config.MSGConfig.critic_random_init=False' \
--gin_bindings='msg.config.MSGConfig.perform_sarsa_q_eval=False' \
--gin_bindings='msg.config.MSGConfig.eval_with_q_filter=False' \
--gin_bindings='msg.config.MSGConfig.num_eval_samples=32' \
--gin_bindings='msg.config.MSGConfig.mimo_using_adamw=False' \
--gin_bindings='msg.config.MSGConfig.mimo_using_obs_tile=False' \
--gin_bindings='msg.config.MSGConfig.mimo_using_act_tile=False'
```
