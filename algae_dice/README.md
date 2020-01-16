# AlgaeDICE

Code for AlgaeDICE as described in `AlgaeDICE: Policy Gradient from Arbitrary
Experience' by Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and
Dale Schuurmans.

Paper available on arXiv [here](https://arxiv.org/abs/1912.02074).

If you use this codebase for your research, please cite the paper:

```
@article{nachum2019algaedice,
  title={AlgaeDICE: Policy Gradient from Arbitrary Experience},
  author={Nachum, Ofir and Dai, Bo and Kostrikov, Ilya and Chow, Yinlam and
      Li, Lihong and Schuurmans, Dale},
  journal={arXiv preprint arXiv:1912.02074},
  year={2019}
}
```

## Basic Commands

Run AlgaeDICE on HalfCheetah:

```
python -m algae_dice.train_eval --logtostderr --save_dir=$HOME/algae/ \
    --env_name=HalfCheetah-v2 --seed=42
```
