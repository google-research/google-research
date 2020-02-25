# Hamiltonian Monte Carlo Swindles

This repository contains the code for the [Hamiltonian Monte Carlo Swindles](https://arxiv.org/abs/2001.05033).

## Setup

From the parent directory, run these commands:

```bash
sudo apt-get install r-base
pip install -r hmc_swindles/requirements.txt
rm -rf probability
git clone --depth 1 https://github.com/tensorflow/probability.git
cp -R probability/discussion discussion
./hmc_swindles/scripts/fetch_datasets.sh
```

## Experiment Pipeline

The experiment proceeds in 4 steps.

1. Training the pre-conditioner

    This will train the preconditioner and place the checkpoints + statistics inside
    the log directory.

    ```bash
    python3 -m hmc_swindles.neutra_runner --mode train --neutra_log_dir /tmp/hmc_swindles/train --hparams "${HPARAMS}"
    ```

2. Tuning the sampler

    This would typically be called by some external optimization framework (not
    included), but you can also do manual tuning. We provide the best HMC
    hyperparameters in the `tune_outputs` directory in case you don't want to do
    this step.

    ```bash
    python3 -m hmc_swindles.neutra_runner --mode objective --checkpoint_log_dir /tmp/hmc_swindles/train --neutra_log_dir /tmp/hmc_swindles/tune --hparams "${HPARAMS}, tune_objective.step_size: 0.1, tune_objective.num_leapfrog_steps: 1"
    ```

3. Benchmarking the sampler

    ```bash
    python3 -m hmc_swindles.neutra_runner --mode benchmark --checkpoint_log_dir /tmp/hmc_swindles/train --neutra_log_dir /tmp/hmc_swindles/eval --tune_outputs_name $(pwd)/hmc_swindles/tune_outputs/${TUNE_OUTPUTS} --hparams "${HPARAMS}"
    ```

4. Evaluating the sampler

    ```bash
    python3 -m hmc_swindles.neutra_runner --mode eval --checkpoint_log_dir /tmp/hmc_swindles/train --neutra_log_dir /tmp/hmc_swindles/eval --tune_outputs_name $(pwd)/hmc_swindles/tune_outputs/${TUNE_OUTPUTS} --hparams "${HPARAMS}"
    ```

## Hyperparameters

### `sparse_logistic_reg.a_neutra`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.a_neutra
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: a_neutra, neutra_experiment.q_base_scale: 0.1"
```

### `sparse_logistic_reg.a_neutra_mala`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.a_neutra_mala
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: a_neutra, neutra_experiment.q_base_scale: 0.1"
```

### `sparse_logistic_reg.a_neutra_rwm`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.a_neutra_rwm
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: a_neutra_rwm, neutra_experiment.q_base_scale: 0.1"
```

### `sparse_logistic_reg.cv_neutra`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.cv_neutra
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: cv_neutra, neutra_experiment.q_base_scale: 0.1"
```

### `sparse_logistic_reg.cv_neutra_mala`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.cv_neutra_mala
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: cv_neutra, neutra_experiment.q_base_scale: 0.1"
```

### `sparse_logistic_reg.cv_neutra_rwm`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.cv_neutra_rwm
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: cv_neutra_rwm, neutra_experiment.q_base_scale: 0.1"
```

### `sparse_logistic_reg.cva_neutra`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.cva_neutra
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: cva_neutra, neutra_experiment.q_base_scale: 0.1"
```

### `sparse_logistic_reg.cva_neutra_mala`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.cva_neutra_mala
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: cva_neutra, neutra_experiment.q_base_scale: 0.1"
```

### `sparse_logistic_reg.cva_neutra_rwm`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.cva_neutra_rwm
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: cva_neutra_rwm, neutra_experiment.q_base_scale: 0.1"
```

### `sparse_logistic_reg.neutra`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.neutra
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: neutra, neutra_experiment.q_base_scale: 0.1"
```

### `sparse_logistic_reg.neutra_mala`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.neutra_mala
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: neutra, neutra_experiment.q_base_scale: 0.1"
```

### `sparse_logistic_reg.neutra_rwm`

```bash
TUNE_OUTPUTS=sparse_logistic_reg.neutra_rwm
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: iaf, iaf_bijector.num_stages: 3, iaf_bijector.hidden_layers: [51, 51], target_spec.name: german_credit_numeric_sparse_logistic_regression, neutra_experiment.mcmc_type: neutra_rwm, neutra_experiment.q_base_scale: 0.1"
```

### `irt.a_neutra`

```bash
TUNE_OUTPUTS=irt.a_neutra
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: a_neutra"
```

### `irt.a_neutra_mala`

```bash
TUNE_OUTPUTS=irt.a_neutra_mala
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: a_neutra"
```

### `irt.a_neutra_rwm`

```bash
TUNE_OUTPUTS=irt.a_neutra_rwm
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: a_neutra_rwm"
```

### `irt.cv_neutra`

```bash
TUNE_OUTPUTS=irt.cv_neutra
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: cv_neutra"
```

### `irt.cv_neutra_mala`

```bash
TUNE_OUTPUTS=irt.cv_neutra_mala
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: cv_neutra"
```

### `irt.cv_neutra_rwm`

```bash
TUNE_OUTPUTS=irt.cv_neutra_rwm
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: cv_neutra_rwm"
```

### `irt.cva_neutra`

```bash
TUNE_OUTPUTS=irt.cva_neutra
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: cva_neutra"
```

### `irt.cva_neutra_mala`

```bash
TUNE_OUTPUTS=irt.cva_neutra_mala
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: cva_neutra"
```

### `irt.cva_neutra_rwm`

```bash
TUNE_OUTPUTS=irt.cva_neutra_rwm
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: cva_neutra_rwm"
```

### `irt.neutra`

```bash
TUNE_OUTPUTS=irt.neutra
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: neutra"
```

### `irt.neutra_mala`

```bash
TUNE_OUTPUTS=irt.neutra_mala
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: neutra"
```

### `irt.neutra_rwm`

```bash
TUNE_OUTPUTS=irt.neutra_rwm
HPARAMS="tune_objective.minibatch_size: 128, eval_mode.batch_size: 128, benchmark.batch_size: 128, neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 4096, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: stan_item_response_theory, neutra_experiment.mcmc_type: neutra_rwm"
```

### `logistic_reg.a_neutra`

```bash
TUNE_OUTPUTS=logistic_reg.a_neutra
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: a_neutra"
```

### `logistic_reg.a_neutra_mala`

```bash
TUNE_OUTPUTS=logistic_reg.a_neutra_mala
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: a_neutra"
```

### `logistic_reg.a_neutra_rwm`

```bash
TUNE_OUTPUTS=logistic_reg.a_neutra_rwm
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: a_neutra_rwm"
```

### `logistic_reg.cv_neutra`

```bash
TUNE_OUTPUTS=logistic_reg.cv_neutra
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: cv_neutra"
```

### `logistic_reg.cv_neutra_mala`

```bash
TUNE_OUTPUTS=logistic_reg.cv_neutra_mala
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: cv_neutra"
```

### `logistic_reg.cv_neutra_rwm`

```bash
TUNE_OUTPUTS=logistic_reg.cv_neutra_rwm
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: cv_neutra_rwm"
```

### `logistic_reg.cva_neutra`

```bash
TUNE_OUTPUTS=logistic_reg.cva_neutra
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: cva_neutra"
```

### `logistic_reg.cva_neutra_mala`

```bash
TUNE_OUTPUTS=logistic_reg.cva_neutra_mala
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: cva_neutra"
```

### `logistic_reg.cva_neutra_rwm`

```bash
TUNE_OUTPUTS=logistic_reg.cva_neutra_rwm
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: cva_neutra_rwm"
```

### `logistic_reg.neutra`

```bash
TUNE_OUTPUTS=logistic_reg.neutra
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: neutra"
```

### `logistic_reg.neutra_mala`

```bash
TUNE_OUTPUTS=logistic_reg.neutra_mala
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: neutra"
```

### `logistic_reg.neutra_rwm`

```bash
TUNE_OUTPUTS=logistic_reg.neutra_rwm
HPARAMS="neutra_experiment.base_learning_rate: 1.0e-2, train.num_steps: 10000, eval_mode.total_batch: 16384, neutra_experiment.learning_rate_schedule: [[2000, 1.0e-1], [8000, 1.0e-2]], neutra_experiment.do_polyak: true, neutra_experiment.polyak_start: 9000, tune_objective.batch_size: 256, tune_objective.f_name: params_var, tune_objective.percentile: 0, neutra_experiment.bijector: affine, affine_bijector.use_tril: true, target_spec.name: german_credit_numeric_logistic_regression, neutra_experiment.mcmc_type: neutra_rwm"
```
