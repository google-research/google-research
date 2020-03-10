# Learned Optimizer List

## What this is

The “Learned Optimizer List” is a sequential list of hyperparameters that have been selected based on performance on a large set of machine learning tasks.

Instead of worrying about finding a good hyperparameter search space for your problem, or complex hyperparameter search methods, we suggest you try using this list of hyperparameters instead.

Wondering how this list is made? See our paper: "Using a thousand optimization tasks to learn hyperparameter search strategies" on [arxiv](https://arxiv.org/abs/2002.11887).

Contact Luke Metz (lmetz@google.com) for questions or issues.

## How to use it

We provide the hyperparameter list as a drop-in optimizer replacement in your favorite machine learning framework (TensorFlow, PyTorch, and Jax).

To use the list of hyperparameters, first figure out how many training steps you want to train your model for,
and second, try multiple different configurations from the list in order (the `idx` argument).
The list is sorted, and we have found a small number of trials, e.g. under 10, should lead to good results, but for best performance continue trying values up to 100.

Try it out on your problem and let us know where it works, and where it doesn't!

We provide full example usage in [examples/](https://github.com/google-research/google-research/tree/master/opt_list/examples) and code snippets below.

### PyTorch
Full example: `python3 -m opt_list.examples.torch`

```python
from opt_list import torch_opt_list

opt = torch_opt_list.optimizer_for_idx(model.parameters(), idx=0,
                                       training_steps=training_steps)
for i in range(training_steps):
  loss = forward()
  opt.zero_grad()
  loss.backward()
  opt.step()
```

### Jax
Full example: `python3 -m opt_list.examples.jax`


For now support Flax style optimizers. It should be fairly easy to port this code
to work with other optimizer apis.
```python
from opt_list import jax_opt_list

optimizer_def = jax_opt_list.optimizer_for_idx(idx=0, training_steps)
optimizer = optimizer_def.create(model)
for i in range(training_steps):
  optimizer, loss = optimizer.optimize(loss_fn)
```

### TF V1
Full example: `python3 -m opt_list.examples.tf_v1`

```python
from opt_list import tf_opt_list

global_step = tf.train.get_or_create_global_step()
opt = tf_opt_list.optimizer_for_idx(0, training_steps, iteration=global_step)
train_op = opt.minimize(loss)

with tf.Session() as sess:
  for i in range(training_iters):
    sess.run([train_op])
```

### TF2.0 / Keras
Full example: `python3 -m opt_list.examples.tf_keras`

```python
from opt_list import tf_opt_list

opt = tf_opt_list.keras_optimizer_for_idx(0, training_steps)
model.compile(loss='mse', optimizer=opt, metrics=[])

for i in range(training_steps):
  model.train_on_batch(inp, target)
```

## Citation

```
@misc{metz2020using,
    title={Using a thousand optimization tasks to learn hyperparameter search strategies},
    author={Luke Metz and Niru Maheswaranathan and Ruoxi Sun and C. Daniel Freeman and Ben Poole and Jascha Sohl-Dickstein},
    year={2020},
    eprint={2002.11887},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


