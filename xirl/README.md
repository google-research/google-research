# xirl

This code accompanies the paper [XIRL: Cross-embodiment Inverse Reinforcement Learning](https://x-irl.github.io/).

If you find this code useful, consider citing our work:

```bibtex
@inproceedings{zakka2021xirl,
    title = {XIRL: Cross-embodiment Inverse Reinforcement Learning},
    author = {Zakka, Kevin and Zeng, Andy and Florence, Pete and Tompson, Jonathan and Bohg, Jeannette and Dwibedi, Debidatta},
    booktitle = {arXiv preprint arXiv:2106.03911},
    year = {2021}
}
```

## Table of Contents

* [Setup]()
* [Datasets]()
* [Experiments]()
    * [Pretraining]()
    * [Policy learning]()
* [Extending XIRL]()

## Experiments: Reproducing Paper Results

### Representation Learning

#### x-MAGICAL

1. Same-embodiment setting

```bash
ALGO="tcn"

# Will train each embodiment sequentially.
python scripts/xmagical_same_embodiment.py --algo=$ALGO
```

1. Cross-embodiment setting

```bash
ALGO="goal_classifier"

# Will train each embodiment sequentially.
python scripts/xmagical_cross_embodiment.py --algo=$ALGO
```

### RLV

WIP

### Policy Learning

#### x-MAGICAL

1. Environment reward

```bash
EMBODIMENT="longstick"
NAME="env_reward_${EMBODIMENT}"

python train_policy.py \
    --experiment_name=$NAME \
    --embodiment=$EMBODIMENT
```

2. Learned reward

```bash
EMBODIMENT="longstick"
ALGO="xirl"
NAME="learned_reward_${EMBODIMENT}_${ALGO}"
MODEL="/tmp/xirl/pretrain_runs/longstick_cross"

python train_policy.py \
    --experiment_name=$NAME \
    --embodiment=$EMBODIMENT \
```

#### RLV

WIP

## Acknowledgments
