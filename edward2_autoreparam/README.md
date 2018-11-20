# Automatic Reparameterisation in Probabilistic Programming

This directory contains code for the paper "Automatic Reparameterisation in
Probabilistic Programming", submitted to [AABI
2018](http://approximateinference.org). This includes interceptors to transform
Edward2 programs into [non-centered](https://arxiv.org/abs/1312.0906) or
partially non-centered parameterizations, code for
interleaved HMC (iHMC) and Variationally Inferred Parameterisation (VIP), and
infrastructure for running experiments, including the models and data used in
the paper.

## Usage

The script `run_experiments.py` is the main entry point. For example, to
evaluate the German credit model with four leapfrog steps per sample, you might
run (from the toplevel `google_research` directory):

```shell
python -m edward2_autoreparam.run_experiments --method=baseline --model=german_credit_lognormalcentered --num_leapfrog_steps=4 --num_mc_samples=16 --num_optimization_steps=2000 --num_samples=50000 --burnin=8000 --num_adaptation_steps=6000 --results_dir=/tmp/results
python -m edward2_autoreparam.run_experiments --method=vip --model=german_credit_lognormalcentered --num_leapfrog_steps=4 --num_mc_samples=16 --num_optimization_steps=2000 --num_samples=50000 --burnin=8000 --num_adaptation_steps=6000 --results_dir=/tmp/results
```

Available options are:

- `method`:
  - `vip`
  - `vip_iaf` (runs VIP with an inverse autoregressive flow
   rather than mean-field normal posterior)
  - `baseline` (runs CP-HMC, NCP-HMC, and iHMC)
- `model`: `radon_stddvs`, `radon`, `german_credit_lognormalcentered`,
  `german_credit_gammascale`, and `8schools`
- `dataset` (used only for radon models): `MA`, `IN`, `PA`, `MO`, `ND`, `MA`,
  or `AZ`

Each run will save results as Python pickle files to the specified directory
`/tmp/results`. To generate human-readable analysis, run

```shell
python -m edward2_autoreparam.analyze_results --results_dir=/tmp/results --model_and_dataset=german_credit_lognormalcentered_na
cat /tmp/results/analysis/german_credit_lognormalcentered_na_analysis.txt
```

## Authors

- [Maria Gorinova](http://homepages.inf.ed.ac.uk/s1207807/) ([m.gorinova@ed.ac.uk](m.gorinova@ed.ac.uk)): primary contact.
- [Dave Moore](http://davmre.github.io) ([davmre@google.com](davmre@google.com))
