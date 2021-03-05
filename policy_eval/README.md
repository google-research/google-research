# Off-policy Evaluation Library

This library was originally written to implement the continuous control
experiments in `Statistical Bootstrapping for Uncertainty Estimation in
Off-Policy Evaluation' by Ilya Kostrikov and Ofir Nachum. Beyond implementing a
bootstrapped version of FQE and MB, which was the focus of that paper, it also
includes a generic implementation of DualDICE as well as weighted per-step
importance sampling and doubly robust estimators.

Paper available on arXiv [here](https://arxiv.org/abs/2007.13609).

If you use this codebase for your research, please cite the paper:

```
@article{kostrikov2020statistical,
         title={Statistical Bootstrapping for Uncertainty Estimation in Off-Policy Evaluation},
         author={Ilya Kostrikov and Ofir Nachum},
         journal={arXiv preprint arXiv:2007.13609},
         year={2020},
}
```

The code allows for generating offline datasets (as used for the original paper)
as well as using D4RL datasets and target policies, as used for the benchmarking
paper `Benchmarks for Deep Off-Policy Evaluation' by Fu, et al.

## Basic Commands

Generate an offline dataset, which uses a behavior policy derived as the target
policy with a hard-coded standard deviation:

```
python -m policy_eval.create_dataset --logtostderr --env_name=Reacher-v2 \
  --models_dir=policy_eval/data
```

Train a vanilla FQE on the dataset. Before training, the code will attempt to
estimate the true value of the target policy using Monte Carlo rollouts. This
can be slow, so to reduce this step, just set --num_mc_episodes to something
small, like 2, although this will affect the reported errors (true minus pred
returns).

```
python -m policy_eval.train_eval --logtostderr --env_name=Reacher-v2 \
  --target_policy_std=0.1 --num_mc_episodes=256 --nobootstrap --algo=fqe \
  --noise_scale=0.0
```

Train FQE on a bootstrapped version of the dataset with reward noise (based on
the Kostrikov, et al. paper above). Note that this will only run a single FQE
training. To actually compute bootstrapped confidence intervals, one would need
to run multiple of these and then aggregate the results according to Efron's
bootstrap computation.

```
python -m policy_eval.train_eval --logtostderr --env_name=Reacher-v2 \
  --target_policy_std=0.1 --num_mc_episodes=256 --bootstrap --algo=fqe \
  --noise_scale=0.25
```

Train a vanilla FQE on a D4RL dataset and policy. To run this, you will need to
download the D4RL policies ([gs://offline-rl/evaluation/d4rl](https://console.cloud.google.com/storage/browser/offline-rl/evaluation/d4rl)) to a suitable location.

```
python -m policy_eval.train_eval --logtostderr --d4rl \
  --env_name=halfcheetah-medium-v0 \
  --d4rl_policy_filename=/path/to/d4rl_policies/halfcheetah/halfcheetah_online_0.pkl \
  --target_policy_std=0.0 --num_mc_episodes=256 --nobootstrap --algo=fqe \
  --noise_scale=0.0
```

This library uses utilities from
[TF-Agents](https://github.com/tensorflow/agents).
