# NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport

This codebase works best if you have a GPU, otherwise you may want to reduce the
batch sizes.

TODO(siege): Add a link to the paper.

## Unconditional Target Distributions

### Funnel

```bash
python -m neutra.neutra_runner --alsologtostderr --neutra_log_dir="/tmp/neutra/funnel_iaf" --mode="all" --hparams="target_spec.name: funnel, target_spec.num_dims: 100, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [100, 100], neutra_experiment.bijector: iaf, tune.min_step_size: 1.0e-4, neutra_experiment.base_learning_rate: 1.0e-2, train_bijector.num_steps: 5000, dense_ar.log_scale_factor: 1.0, target_spec.eig_source: gamma, tune.max_step_size: 5.0, tune.max_leapfrog_steps: 100, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[1000, 1.0e-1], [4000, 1.0e-2]], tune.obj_type: rhat, tune.percentile: 0"
```

```bash
python -m neutra.neutra_runner --alsologtostderr --neutra_log_dir="/tmp/neutra/funnel_tril" --mode="all" --hparams="target_spec.name: funnel, target_spec.num_dims: 100, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [100, 100], neutra_experiment.bijector: affine, tune.min_step_size: 1.0e-4, neutra_experiment.base_learning_rate: 1.0e-2, train_bijector.num_steps: 5000, dense_ar.log_scale_factor: 1.0, target_spec.eig_source: gamma, tune.max_step_size: 5.0, tune.max_leapfrog_steps: 100, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[1000, 1.0e-1], [4000, 1.0e-2]], tune.obj_type: rhat, tune.percentile: 0, affine_bijector.use_tril: true"
```

```bash
python -m neutra.neutra_runner --alsologtostderr --neutra_log_dir="/tmp/neutra/funnel_diag" --mode="all" --hparams="target_spec.name: funnel, target_spec.num_dims: 100, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [100, 100], neutra_experiment.bijector: affine, tune.min_step_size: 1.0e-4, neutra_experiment.base_learning_rate: 1.0e-2, train_bijector.num_steps: 5000, dense_ar.log_scale_factor: 1.0, target_spec.eig_source: gamma, tune.max_step_size: 5.0, tune.max_leapfrog_steps: 100, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[1000, 1.0e-1], [4000, 1.0e-2]], tune.obj_type: rhat, tune.percentile: 0"
```

### Ill-Conditioned Gaussian

```bash
python -m neutra.neutra_runner --alsologtostderr --neutra_log_dir="/tmp/neutra/ill_cond_gaussian_iaf" --mode="all" --hparams="target_spec.name: new_ill_cond_gaussian, target_spec.num_dims: 100, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [100, 100], neutra_experiment.bijector: iaf, tune.min_step_size: 1.0e-4, neutra_experiment.base_learning_rate: 1.0e-2, train_bijector.num_steps: 5000, dense_ar.log_scale_factor: 1.0, target_spec.eig_source: gamma, target_spec.gamma_shape: 0.8, tune.max_step_size: 5.0, tune.max_leapfrog_steps: 100, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[1000, 1.0e-1], [4000, 1.0e-2]], tune.obj_type: rhat, tune.percentile: 0"
```

```bash
python -m neutra.neutra_runner --alsologtostderr --neutra_log_dir="/tmp/neutra/ill_cond_gaussian_tril" --mode="all" --hparams="target_spec.name: new_ill_cond_gaussian, target_spec.num_dims: 100, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [100, 100], neutra_experiment.bijector: affine, tune.min_step_size: 1.0e-4, neutra_experiment.base_learning_rate: 1.0e-2, train_bijector.num_steps: 5000, dense_ar.log_scale_factor: 1.0, target_spec.eig_source: gamma, target_spec.gamma_shape: 0.8, tune.max_step_size: 5.0, tune.max_leapfrog_steps: 100, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[1000, 1.0e-1], [4000, 1.0e-2]], tune.obj_type: rhat, tune.percentile: 0, affine_bijector.use_tril: true"
```

```bash
python -m neutra.neutra_runner --alsologtostderr --neutra_log_dir="/tmp/neutra/ill_cond_gaussian_diag" --mode="all" --hparams="target_spec.name: new_ill_cond_gaussian, target_spec.num_dims: 100, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [100, 100], neutra_experiment.bijector: affine, tune.min_step_size: 1.0e-4, neutra_experiment.base_learning_rate: 1.0e-2, train_bijector.num_steps: 5000, dense_ar.log_scale_factor: 1.0, target_spec.eig_source: gamma, target_spec.gamma_shape: 0.8, tune.max_step_size: 5.0, tune.max_leapfrog_steps: 100, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[1000, 1.0e-1], [4000, 1.0e-2]], tune.obj_type: rhat, tune.percentile: 0"
```

### Sparse Logistic Regression

For this task you need to grab the German Credit dataset from
https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric.

We also first run Stan to get the ground truth values:

```bash
python -m neutra.logistic_reg_gamma_pystan
```

And then we can run the NeuTra experiments:

```bash
python -m neutra.neutra_runner --alsologtostderr --neutra_log_dir="/tmp/neutra/ill_cond_gaussian_diag" --mode="all" --hparams="target_spec.name: logistic_reg, target_spec.num_dims: 100, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], neutra_experiment.bijector: affine, tune.min_step_size: 1.0e-4, neutra_experiment.base_learning_rate: 1.0e-2, train_bijector.num_steps: 5000, dense_ar.log_scale_factor: 1.0, target_spec.eig_source: gamma, target_spec.gamma_shape: 1.0, tune.max_step_size: 5.0, tune.max_leapfrog_steps: 100, target_spec.precomputed_stats_path: /tmp/logistic_gamma_0, german.path: /tmp/german.data-numeric, german.numeric: true, target_spec.regression_type: gamma_scales2, target_spec.regression_normalize: true, target_spec.regression_dataset: german, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[1000, 1.0e-1], [4000, 1.0e-2]], tune.obj_type: rhat, tune.percentile: 0, affine_bijector.use_tril: true"
```

```bash
python -m neutra.neutra_runner --alsologtostderr --neutra_log_dir="/tmp/neutra/ill_cond_gaussian_diag" --mode="all" --hparams="target_spec.name: logistic_reg, target_spec.num_dims: 100, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], neutra_experiment.bijector: affine, tune.min_step_size: 1.0e-4, neutra_experiment.base_learning_rate: 1.0e-2, train_bijector.num_steps: 5000, dense_ar.log_scale_factor: 1.0, target_spec.eig_source: gamma, target_spec.gamma_shape: 1.0, tune.max_step_size: 5.0, tune.max_leapfrog_steps: 100, target_spec.precomputed_stats_path: /tmp/logistic_gamma_0, german.path: /tmp/german.data-numeric, german.numeric: true, target_spec.regression_type: gamma_scales2, target_spec.regression_normalize: true, target_spec.regression_dataset: german, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[1000, 1.0e-1], [4000, 1.0e-2]], tune.obj_type: rhat, tune.percentile: 0"
```

```bash
python -m neutra.neutra_runner --alsologtostderr --neutra_log_dir="/tmp/neutra/ill_cond_gaussian_diag" --mode="all" --hparams="target_spec.name: logistic_reg, target_spec.num_dims: 100, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], neutra_experiment.bijector: iaf, tune.min_step_size: 1.0e-4, neutra_experiment.base_learning_rate: 1.0e-2, train_bijector.num_steps: 5000, dense_ar.log_scale_factor: 1.0, target_spec.eig_source: gamma, target_spec.gamma_shape: 1.0, tune.max_step_size: 5.0, tune.max_leapfrog_steps: 100, target_spec.precomputed_stats_path: /tmp/logistic_gamma_0, german.path: /tmp/german.data-numeric, german.numeric: true, target_spec.regression_type: gamma_scales2, target_spec.regression_normalize: true, target_spec.regression_dataset: german, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[1000, 1.0e-1], [4000, 1.0e-2]], neutra_experiment.q_base_scale: 0.1, tune.obj_type: rhat, tune.percentile: 0"
```

## Conditional Target Distributions

For these experiments you need to obtain the MNIST dataset.

### Diag VAE

```bash
python -m neutra.vae --mnist_data_dir=/tmp --alsologtostderr --mode=train --model=vae --train_dir=/tmp/neutra/vae_diag/train --eval_dir=/tmp/neutra/vae_diag/eval --hparams="vae.beta_steps: 24000, vae.bijector_type: iaf, vae.condition_bijector: true, dense_recognition.sigma_activation: softplus, dense_recognition_iaf.sigma_activation: softplus, vae.z_dims: 32, vae.encoder_type: conv, dense_recognition_iaf.iaf_layer_sizes: [1920, 1920], learning_rate.schedule: new, train.epochs: 1000"

python -m neutra.vae --mnist_data_dir=/tmp --alsologtostderr --mode=ais_eval --model=vae --ais_num_chains 20 --train_dir=/tmp/neutra/vae_diag/train --eval_dir=/tmp/neutra/vae_diag/eval --hparams="vae.beta_steps: 24000, vae.bijector_type: iaf, vae.condition_bijector: true, dense_recognition.sigma_activation: softplus, dense_recognition_iaf.sigma_activation: softplus, vae.z_dims: 32, vae.encoder_type: conv, dense_recognition_iaf.iaf_layer_sizes: [1920, 1920], learning_rate.schedule: new, train.epochs: 1000, ais.step_size: 0.05"
```

### IAF VAE

```bash
python -m neutra.vae --mnist_data_dir=/tmp --alsologtostderr --mode=train --model=vae --train_dir=/tmp/neutra/vae_iaf/train --eval_dir=/tmp/neutra/vae_iaf/eval --hparams="vae.beta_steps: 24000, vae.bijector_type: iaf, vae.condition_bijector: true, dense_recognition.sigma_activation: softplus, dense_recognition_iaf.sigma_activation: softplus, vae.z_dims: 32, vae.encoder_type: conv, dense_recognition_iaf.iaf_layer_sizes: [1920, 1920], learning_rate.schedule: new, train.epochs: 1000"

python -m neutra.vae --mnist_data_dir=/tmp --alsologtostderr --mode=ais_eval --model=vae --ais_num_chains 20 --train_dir=/tmp/neutra/vae_iaf/train --eval_dir=/tmp/neutra/vae_iaf/eval --hparams="vae.beta_steps: 24000, vae.bijector_type: iaf, vae.condition_bijector: true, dense_recognition.sigma_activation: softplus, dense_recognition_iaf.sigma_activation: softplus, vae.z_dims: 32, vae.encoder_type: conv, dense_recognition_iaf.iaf_layer_sizes: [1920, 1920], ais.step_size: 0.05"
```

### IAF VAE + NeuTra

```bash
python -m neutra.vae --mnist_data_dir=/tmp --alsologtostderr --mode=train --model=dlgm --train_dir=/tmp/neutra/vae_neutra/train --eval_dir=/tmp/neutra/vae_neutra/eval --hparams="dlgm.beta_steps: 24000, dlgm.bijector_type: iaf, dlgm.condition_bijector: true, dense_recognition.sigma_activation: softplus, dense_recognition_iaf.sigma_activation: softplus, dlgm.z_dims: 32, dlgm.encoder_type: conv, dense_recognition_iaf.iaf_layer_sizes: [1920, 1920], dlgm.step_size: 0.1, dlgm.num_hmc_steps: 1, dlgm.num_leapfrog_steps: 4, dlgm.no_gen_train_steps: 3000, learning_rate.schedule: new, train.epochs: 1000"

python -m neutra.vae --mnist_data_dir=/tmp --alsologtostderr --mode=ais_eval --model=dlgm --ais_num_chains 20 --train_dir=/tmp/neutra/vae_neutra/train --eval_dir=/tmp/neutra/vae_neutra/eval --hparams="dlgm.beta_steps: 24000, dlgm.bijector_type: iaf, dlgm.condition_bijector: true, dense_recognition.sigma_activation: softplus, dense_recognition_iaf.sigma_activation: softplus, dlgm.z_dims: 32, dlgm.encoder_type: conv, dense_recognition_iaf.iaf_layer_sizes: [1920, 1920], dlgm.step_size: 0.1, dlgm.num_hmc_steps: 1, dlgm.num_leapfrog_steps: 4, dlgm.no_gen_train_steps: 3000, ais.step_size: 0.05"
```

## Authors

- Matt Hoffman (mhoffman@google.com)
- Pavel Sountsov (siege@google.com)
