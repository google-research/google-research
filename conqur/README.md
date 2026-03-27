# ConQUR

Code for ConQUR as described in "ConQUR: Mitigating Delusional Bias in Deep
Q-learning" by Andy Su, Jayden Ooi, Tyler Lu, Dale Schuurmans, and Craig
Boutilier.

The paper is available on arXiv [here](https://arxiv.org/abs/2002.12399).


## Pretrained DQN checkpoint

https://console.cloud.google.com/storage/browser/download-dopamine-rl/lucid/dqn/


## Basic Command

To run ConQUR on Pong without pretrained checkpoint:

```
python -m conqur.main --save_dir=$HOME/conqur --env_name=Pong --logtostderr
```

To run ConQUR on Pong with pretrained checkpoint:

```
python -m conqur.main --save_dir=$HOME/conqur --env_name=Pong --logtostderr ----pretrain_model_path=/Users/DiJia/Desktop/icml_conqur/checkpoint/Pong/lucid_dqn_Pong_2_tf_ckpt-199
```

