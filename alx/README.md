# ALX: Large Scale Matrix Factorization on TPUs

We present ALX, an open-source library for distributed matrix factorization
using Alternating Least Squares, written in JAX. Our design allows for efficient
use of the TPU architecture and scales well to matrix factorization problems of
O(B) rows/columns by scaling the number of available TPU cores.

More details here: https://arxiv.org/abs/2112.02194v1

# Usage

A rudimentary structure of how this library can be used:

```python
ds, tds, test_ds = dataset_utils.build_datasets(cfg=cfg)

# Initialize model_dir and setup summary writer.
if jax.process_index() == 0:
  tf.io.gfile.makedirs(FLAGS.model_dir)

summary_writer = tensorboard.SummaryWriter(
    os.path.join(FLAGS.model_dir, 'eval'))
summarize_gin_config(model_dir=FLAGS.model_dir, summary_writer=summary_writer)

# Check if there are any intermediate checkpoints.
state = checkpoints.restore_checkpoint(FLAGS.model_dir)
als_state = None
if state:
  als_state = als.ALSState(**state)

model = als.ALS(cfg=cfg, als_state=als_state)
for epoch in range(model.als_state.step, cfg.num_epochs):
  model.train(ds, tds)

  # Save a checkpoint after every epoch.
  checkpoints.save_checkpoint(model.als_state, FLAGS.model_dir)
  metrics = model.eval(test_ds)
  if jax.process_index() == 0:
    for key, val in zip([
        f'Recall@20/{jax.process_index()}',
        f'Recall@50/{jax.process_index()}',
        f'Num valid examples/{jax.process_index()}'
    ], list(metrics)):
      summary_writer.scalar(key, val, epoch)
    logging.info(str(metrics))
```

## License

Licensed under the Apache 2.0 License.

## Disclaimer

This is not an officially supported Google product.
