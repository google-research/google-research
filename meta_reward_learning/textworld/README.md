# Google AI Research

This repository contains code released by
[Google AI Research](https://ai.google/research) for the instruction following
task described in the paper
[Learning to Generalize from Sparse and Underspecified Rewards](https://arxiv.org/abs/1902.07198).

Download the datasets from this [url](https://storage.googleapis.com/merl/textworld/datasets.tar.gz)
to `$TEXTWORLD_LOCATION`.

After downloading the training data, you can run the training via running
the `experiment.py` file:

```
python -m meta_reward_learning.textworld.experiment\
            --test_file=$TEXTWORLD_LOCATION/datasets/textworld-test.pkl\
            --train_file=$TEXTWORLD_LOCATION/datasets/textworld-train.pkl\
            --dev_file=$TEXTWORLD_LOCATION/datasets/textworld-dev.pkl\
            --n_train_envs=240 --n_dev_envs=60 --log_summaries
```

`common_flags.py` contains all the flags which can be passed to `experiment.py`

`use_gold_trajs` specifies whether you are using gold trajectories or not.
Turn it on using `--use_gold_trajs` for the Oracle reward setup.
To use MeRL, specify the flag `--meta_learn` along with the hyperparameter
values specified in the paper.

---

*Disclaimer: This is not an official Google product.*
