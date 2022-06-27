# SNR

Note: The most important file in `snr` is `snr_utils.py` which contains utilities
for playing with the Spectral Norm Regularization research ideas we had. They
are utilities that can be added on top of any other agent. The agent in this
directory is almost like a vanilla actor-critic method with SNR on top.

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
`snr.config.SNRConfig.num_bc_iters` and `snr.config.SNRConfig.pretrain_iters`
are set in terms of "true" number of steps, i.e. no need to account for
`num_sgd_steps_per_step`.
For easier local debugging, you can set:
```
--num_sgd_steps_per_step 1 \
--batch_size 64 \
--num_steps 1000 \
--episodes_per_eval 10 \
--gin_bindings='snr.config.SNRConfig.num_sgd_steps_per_step=1'
```

Note: For additional parameters that can be set please refer to `snr/config.py`

## Running the SNR agent
```
python3 -m jrl.localized.runner \
--pdb_post_mortem \
--debug_nans=False \
--create_saved_model_actor=False \
--num_steps 11000 \
--eval_every_steps 500 \
--episodes_per_eval 100 \
--batch_size 51200 \
--root_dir '/tmp/test_snr' \
--seed 42 \
--algorithm 'snr' \
--task_class 'd4rl' \
--task_name 'antmaze-large-diverse-v0' \
--gin_bindings='snr.config.SNRConfig.num_sgd_steps_per_step=200' \
--gin_bindings='snr.config.SNRConfig.num_bc_iters=50000' \
--gin_bindings='snr.config.SNRConfig.actor_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='snr.config.SNRConfig.critic_network_hidden_sizes=(256, 256, 256)' \
--gin_bindings='snr.config.SNRConfig.num_critics=1' \
--gin_bindings='snr.config.SNRConfig.use_snr_in_bc_iters=False' \
--gin_bindings='snr.config.SNRConfig.snr_applied_to="policy"' \
--gin_bindings='snr.config.SNRConfig.snr_alpha=10' \
--gin_bindings='snr.config.SNRKwargs.snr_mode="params_kernel"' \
--gin_bindings='snr.config.SNRKwargs.snr_loss_type="svd_kamyar_v1"' \
--gin_bindings='snr.config.SNRKwargs.snr_num_centroids=2048' \
--gin_bindings='snr.config.SNRKwargs.snr_kmeans_iters=1' \
--gin_bindings='snr.config.SNRKwargs.use_log_space_matrix=False' \
--gin_bindings='snr.config.SNRKwargs.snr_matrix_tau=0.01' \
--gin_bindings='snr.config.SNRConfig.snr_kwargs=@snr.config.SNRKwargs()'
```
