# Datasets

## `jrl/data/__init__.py`
An example of how the data loader should be setup can be found at
`jrl/data/d4rl.py` for the D4RL case. The data
iterator should be added to `jrl/data/__init__.py`. The key things to note are
the data format being
`Inputs(data=(states, actions, rewards, discounts, next_states, next_actions))`.
`next_actions` is not needed in many algorithms (for example in MSG, we need
`next_actions` if `use_sass=True` or `num_q_repr_pretrain_iters` is greater
than 0).
Note to add `return dataset.as_numpy_iterator()` when creating your dataloaders.
