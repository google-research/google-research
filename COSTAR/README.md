# COSTAR: COunterfactual Self-supervised TrAnsformeR 

[![arXiv](https://img.shields.io/badge/arXiv-2311.00886-b31b1b.svg)](https://arxiv.org/abs/2311.00886)

## Environment Setup

```bash
conda env create -f environment.yml
conda activate costar
# For compatibility with saved data
conda install -c conda-forge pandas==1.5.3

# Setup wandb for logging experiments
wandb login
```

## Dataset Setup

Download the
[pre-generated data](https://zenodo.org/record/8412226/files/all_needed_data.tar.gz)
and extract in the project root directory:

```bash
tar xzvf all_needed_data.tar.gz
```

To add customized datasets, follow instructions specified [here](dataset_instructions.md).

### Tumor growth

No further setup needed. All data will be generated on the fly.

### Semi-synthetic MIMIC-III

No further setup needed. The pre-generated data will be loaded. To re-run the
simulation process (can be slow), use:

```bash
PYTHONPATH=. python runnables/train_enc_dec.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=crn_noncausal_troff +'backbone/crn_noncausal_troff_hparams/mimic3_synthetic="all"' dataset.data_gen_n_jobs=8 exp.gen_data_only=True exp.seed=17,43,44,91,95 exp.tags=230814_mimicsyn_all_data
PYTHONPATH=. python runnables/train_enc_dec.py -m +dataset/mimic3_syn_age="0-3_all_srctest" +backbone=crn_noncausal_troff +'backbone/crn_noncausal_troff_hparams/mimic3_synthetic="all"' dataset.data_gen_n_jobs=8 exp.gen_data_only=True exp.seed=17,43,44,91,95 exp.tags=230910_mimicsyn_all_data
```

### M5

No further setup needed. All data will be generated on the fly.

## Experiments

We list the detailed commands of running experiments and metrics to report in
the following script files.

### COSTAR

See [`scripts_release/costar.sh`](scripts_release/costar.sh) for details.

### Baselines

See [`scripts_release/baselines.sh`](scripts_release/baselines.sh) for details.
