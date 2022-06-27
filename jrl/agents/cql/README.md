# CQL

Note: In the MSG paper we reported D4RL gym results using this implementation as well
as another implementation that was not ours. For D4RL gym this implementation
seems to be good at reproducing CQL results, but were not able to reproduce
the CQL antmaze results (although we and colleagues were unable to reproduce
them with other available implementations as well).

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
`cql.config.CQLConfig.num_bc_iters` and `cql.config.CQLConfig.pretrain_iters`
are set in terms of "true" number of steps, i.e. no need to account for
`num_sgd_steps_per_step`.
For easier local debugging, you can set:
```
--num_sgd_steps_per_step 1 \
--batch_size 64 \
--num_steps 1000 \
--episodes_per_eval 10 \
--gin_bindings='cql.config.CQLConfig.num_sgd_steps_per_step=1'
```

Note: Only for `halfcheetah, hopper, walker` experiments we set
`cql.config.CQLConfig.num_critics=2`. It was not a noticeable difference
but we did not go back to rerun full experiments with this param set to True.
For `antmaze` domains we tried both `num_critics` being 1 or 2. The SNR
hyperparameters are unrelated to CQL and are from orthogonal research ideas.

Note: For additional parameters that can be set please refer to `cql/config.py`

## Running CQL
```
python3 -m jrl.localized.runner \
--pdb_post_mortem \
--debug_nans=False \
--create_saved_model_actor=False \
--num_steps 11000 \
--eval_every_steps 500 \
--episodes_per_eval 100 \
--batch_size 51200 \
--root_dir '/tmp/test_cql' \
--seed 42 \
--algorithm 'cql' \
--task_class 'd4rl' \
--task_name 'antmaze-large-diverse-v0' \
--gin_bindings='cql.config.CQLConfig.num_sgd_steps_per_step=200' \
--gin_bindings='cql.config.CQLConfig.num_bc_iters=50000' \
--gin_bindings='cql.config.CQLConfig.cql_alpha=0.05' \
--gin_bindings='cql.config.CQLConfig.num_importance_acts=10' \
--gin_bindings='cql.config.CQLConfig.actor_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='cql.config.CQLConfig.critic_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='cql.config.CQLConfig.num_critics=2' \
--gin_bindings='cql.config.CQLConfig.tau=0.005' \
--gin_bindings='cql.config.CQLConfig.eval_with_q_filter=False' \
--gin_bindings='cql.config.CQLConfig.num_eval_samples=32' \
--gin_bindings='cql.config.CQLConfig.snr_kwargs=@snr.config.SNRKwargs()' \
--gin_bindings='cql.config.CQLConfig.snr_alpha=0' \
--gin_bindings='snr.config.SNRKwargs.snr_mode="params_kernel"' \
--gin_bindings='snr.config.SNRKwargs.snr_loss_type="svd_kamyar_v1"' \
--gin_bindings='snr.config.SNRKwargs.use_log_space_matrix=False'
```
