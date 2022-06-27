## Implementation of Mazoure et al., Improving Zero-shot Generalization in
## Offline Reinforcement Learning using Generalized Value Functions

Arxiv link to coming soon!

Use [representation_batch_rl/train_eval_offline.py](representation_batch_rl/train_eval_offline.py) and [representation_batch_rl/train_eval_online.py](representation_batch_rl/train_eval_online.py) to execute experiments.

```
cd <path_to>/representation_batch_rl
PYTHONPATH="$(pwd)/.." python representation_batch_rl/train_eval_offline \
  --alsologtostderr \
  --save_dir=/tmp/representation_batch_rl/ \
  --env_name="procgen-bigfish-200-0" \
  --ckpt_timesteps=25_000_000 \
  --max_timesteps=5_000_000 \
  --algo_name=ours \
  --dataset_size=5_000_000 \
  --eval_interval=100 \
  --num_data_augs=1 \
  --pretrain=2 \
  --rep_learn_keywords=linear_Q__popart__cce \
  --save_interval=10 \
  --batch_size=1024 \
  --temp=0.5 \
  --n_quantiles=7
```

Use [representation_batch_rl/make_dataset.py](representation_batch_rl/make_dataset.py) to generate tfrecords for use in experiments.

For more information contact: tompson@google.com or ofirnachum@google.com.
