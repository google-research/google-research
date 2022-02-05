"""Data loading utilities.
"""
import jax.numpy as jnp
import tensorflow_datasets as tfds


NUM_TRAIN = {
  'mnist': 50000,
  'fashion_mnist': 50000,
  'cifar10': 40000,
  'cifar100': 45000
}

means = {
  'mnist': 0.13066,
  'fashion_mnist': 0.28604,
  'cifar10': jnp.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
  'cifar100': jnp.array([x / 255.0 for x in [125.3, 123.0, 113.9]])
}

stds = {
  'mnist': 0.30811,
  'fashion_mnist': 0.35302,
  'cifar10': jnp.array([x / 255.0 for x in [63.0, 62.1, 66.7]]),
  'cifar100': jnp.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
}


def load_data(name):
  """Loads train, validation, and test data for a given dataset name.

  Args:
    name: A string specifying which dataset to load. Can be 'mnist', 'cifar10',
          'cifar100'

  Returns:
    A dictionary containing the inputs and targets for the training set,
    validation set, and test set of the selected dataset.
  """
  train_dataset = tfds.load(name, batch_size=-1, split='train')

  train_images = tfds.as_numpy(train_dataset['image'])[:NUM_TRAIN[name]] / 255.0
  train_labels = tfds.as_numpy(train_dataset['label'])[:NUM_TRAIN[name]]

  val_images = tfds.as_numpy(train_dataset['image'])[NUM_TRAIN[name]:] / 255.0
  val_labels = tfds.as_numpy(train_dataset['label'])[NUM_TRAIN[name]:]

  test_dataset = tfds.load(name, batch_size=-1, split='test')
  all_test_images = tfds.as_numpy(test_dataset['image']) / 255.0
  all_test_labels = tfds.as_numpy(test_dataset['label'])

  train_data = jnp.array((train_images - means[name]) / stds[name])
  train_data = train_data.transpose((0,3,1,2))
  train_targets = jnp.array(train_labels)

  val_data = jnp.array((val_images - means[name]) / stds[name])
  val_data = val_data.transpose((0,3,1,2))
  val_targets = jnp.array(val_labels)

  test_data = jnp.array((all_test_images - means[name]) / stds[name])
  test_data = test_data.transpose((0,3,1,2))
  test_targets = jnp.array(all_test_labels)

  return {'train_data': train_data,
          'train_targets': train_targets,
          'val_data': val_data,
          'val_targets': val_targets,
          'test_data': test_data,
          'test_targets': test_targets}
