# BC

Implementation for Behavioral Cloning (BC).

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
`bc.config.BCConfig.num_bc_iters` and `bc.config.BCConfig.pretrain_iters`
are set in terms of "true" number of steps, i.e. no need to account for
`num_sgd_steps_per_step`.
For easier local debugging, you can set:
```
--num_sgd_steps_per_step 1 \
--batch_size 64 \
--num_steps 1000 \
--episodes_per_eval 10 \
--gin_bindings='bc.config.BCConfig.num_sgd_steps_per_step=1'
```

Note: For `bc.config.BCConfig.loss_type` MSE loss works better than MLE loss
and vice-verse

Note: For additional parameters that can be set please refer to `bc/config.py`

## Running BC
```
python3 -m jrl.localized.runner \
--pdb_post_mortem \
--debug_nans=False \
--create_saved_model_actor=False \
--num_steps 11000 \
--eval_every_steps 500 \
--episodes_per_eval 100 \
--batch_size 51200 \
--root_dir '/tmp/test_bc' \
--seed 42 \
--algorithm 'bc' \
--task_class 'd4rl' \
--task_name 'antmaze-large-diverse-v0' \
--gin_bindings='bc.config.BCConfig.num_sgd_steps_per_step=200' \
--gin_bindings='bc.config.BCConfig.policy_lr=1e-4' \
--gin_bindings='bc.config.BCConfig.loss_type="MLE"' \
--gin_bindings='bc.config.BCConfig.entropy_regularization_weight=0'
```
