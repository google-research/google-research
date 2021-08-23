# xirl

- [Overview](#overview)
- [Setup](#setup)
- [Datasets](#datasets)
- [Experiments: Reproducing Paper Results](#experiments-reproducing-paper-results)
- [Acknowledgments](#acknowledgments)

## Todos

1. x-MAGICAL same-embodiment
  * [x] Env reward
  * [ ] XIRL
    * [x] longstick
    * [x] mediumstick
    * [ ] shortstick
    * [ ] gripper
  * [ ] TCN
  * [ ] Goal classifier
  * [ ] Imagenet
  * [ ] LIFS
  * [ ] SimCLR
2. x-MAGICAL cross-embodiment
  * [x] Env reward
  * [ ] XIRL
    * [ ] longstick
    * [ ] mediumstick
    * [ ] shortstick
    * [ ] gripper
  * [ ] TCN
  * [ ] Goal classifier
  * [ ] Imagenet
  * [ ] LIFS
  * [ ] SimCLR / Moco
3. RLV
  * State pusher
    * [ ] Env reward
    * [ ] XIRL
  * Pixel drawer opening
    * [ ] Env reward
    * [ ] XIRL

* [ ] Investigate `shuffle` and `drop_last` in pretrain dataloaders.

## Overview

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

## Setup

## Datasets

## Experiments: Reproducing Paper Results

|           | Representation Learning | Reinforcement Learning |
| --------- | ----------------------- | ---------------------- |
| x-MAGICAL |                         |                        |
| RLV       |                         |                        |

## Acknowledgments
