# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Helper model functions for retrieving model predictions."""

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from pruning_identified_exemplars.utils import class_level_metrics
from pruning_identified_exemplars.utils import data_input
from pruning_identified_exemplars.utils import resnet_model


def compute_lr(current_epoch, initial_learning_rate, train_batch_size,
               lr_schedule):
  """Computes learning rate schedule."""
  scaled_lr = initial_learning_rate * (train_batch_size / 256.0)

  decay_rate = (
      scaled_lr * lr_schedule[0][0] * current_epoch / lr_schedule[0][1])
  for mult, start_epoch in lr_schedule:
    decay_rate = tf.where(current_epoch < start_epoch, decay_rate,
                          scaled_lr * mult)
  return decay_rate


def train_function(params, loss):
  """Creates the training op that will be optimized by the estimator."""

  global_step = tf.train.get_global_step()

  steps_per_epoch = params["num_train_images"] / params["train_batch_size"]
  current_epoch = (tf.cast(global_step, tf.float32) / steps_per_epoch)
  learning_rate = compute_lr(current_epoch, params["base_learning_rate"],
                             params["train_batch_size"], params["lr_schedule"])
  optimizer = tf.train.MomentumOptimizer(
      learning_rate=learning_rate,
      momentum=params["momentum"],
      use_nesterov=True)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops), tf.name_scope("train"):
    train_op = optimizer.minimize(loss, global_step)

  if params["pruning_method"]:
    pruning_params_string = params["pruning_dict"]
    # Parse pruning hyperparameters
    pruning_hparams = tf.contrib.model_pruning.get_pruning_hparams().parse(
        pruning_params_string)

    # Create a pruning object using the pruning hyperparameters
    pruning_obj = tf.contrib.model_pruning.pruning.Pruning(
        pruning_hparams, global_step=global_step)

    # We override the train op to also update the mask.
    with tf.control_dependencies([train_op]):
      train_op = pruning_obj.conditional_mask_update_op()

    masks = tf.contrib.model_pruning.get_masks()

  with tf2.summary.create_file_writer(params["output_dir"]).as_default():
    with tf2.summary.record_if(True):
      tf2.summary.scalar("loss", loss, step=global_step)
      tf2.summary.scalar("learning_rate", learning_rate, step=global_step)
      tf2.summary.scalar("current_epoch", current_epoch, step=global_step)
      tf2.summary.scalar("steps_per_epoch", steps_per_epoch, step=global_step)
      tf2.summary.scalar(
          "weight_decay", params["weight_decay"], step=global_step)
      if params["pruning_method"]:
        tf2.summary.scalar("pruning_masks", masks, step=global_step)

    tf.summary.all_v2_summary_ops()

  return train_op


def model_fn_w_pruning(features, labels, mode, params):
  """The model_fn for ResNet-50 with pruning.

  Args:
    features: A float32 batch of images.
    labels: A int32 batch of labels.
    mode: Specifies whether training or evaluation.
    params: parameters passed to the eval function.

  Returns:
    A EstimatorSpec for the model
  """

  images = features["image_raw"]
  labels = features["label"]

  if params["task"] in [
      "pie_dataset_gen", "imagenet_predictions", "robustness_imagenet_c",
      "robustness_imagenet_a", "ckpt_prediction"
  ]:
    human_labels = features["human_label"]

  mean_rgb = params["mean_rgb"]
  stddev_rgb = params["stddev_rgb"]

  # Normalize the image to zero mean and unit variance.
  images -= tf.constant(mean_rgb, shape=[1, 1, 3], dtype=images.dtype)
  images /= tf.constant(stddev_rgb, shape=[1, 1, 3], dtype=images.dtype)

  network = resnet_model.resnet_50(
      num_classes=1000,
      pruning_method=params["pruning_method"],
      data_format="channels_last")

  logits = network(
      inputs=images, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
  one_hot_labels = tf.one_hot(labels, params["num_label_classes"])

  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=one_hot_labels,
      label_smoothing=params["label_smoothing"])

  # Add weight decay to the loss for non-batch-normalization variables.
  loss = cross_entropy + params["weight_decay"] * tf.add_n([
      tf.nn.l2_loss(v)
      for v in tf.trainable_variables()
      if "batch_normalization" not in v.name
  ])

  # we run predictions on gpu since ordering is very important and
  # thus we need to run with batch size 1 (not enabled on tpu)
  if mode == tf.estimator.ModeKeys.PREDICT:
    train_op = None
    eval_metrics = None
    predicted_probability = tf.cast(
        tf.reduce_max(tf.nn.softmax(logits, name="softmax"), axis=1),
        tf.float32)

    _, top_5_indices = tf.nn.top_k(tf.to_float(logits), k=5)

    predictions = {
        "predictions": tf.argmax(logits, axis=1),
        "true_class": labels,
        "predicted_probability": predicted_probability,
        "top_5_indices": top_5_indices
    }

  if mode == tf.estimator.ModeKeys.TRAIN:
    train_op = train_function(params, loss)
    eval_metrics = None
    predictions = None

  if mode == tf.estimator.ModeKeys.EVAL:
    train_op = None
    predictions = None
    params_eval = {
        "num_label_classes": params["num_label_classes"],
        "log_class_level_summaries": False
    }
    eval_metrics = class_level_metrics.create_eval_metrics(
        labels, logits, human_labels, params_eval)

  return tf.estimator.EstimatorSpec(
      predictions=predictions,
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics)


def initiate_task_helper(model_params,
                         ckpt_directory=None,
                         pruning_params=None):
  """Get all predictions for eval.

  Args:
    model_params:
    ckpt_directory: model checkpoint directory containing event file
    pruning_params:

  Returns:
    pd.DataFrame containing metrics from event file
  """

  if model_params["task"] != "imagenet_training":
    classifier = tf.estimator.Estimator(
        model_fn=model_fn_w_pruning, params=model_params)

    if model_params["task"] in ["imagenet_predictions"]:
      predictions = classifier.predict(
          input_fn=data_input.input_fn, checkpoint_path=ckpt_directory)
      return predictions

    if model_params["task"] in [
        "robustness_imagenet_a", "robustness_imagenet_c", "robustness_pie",
        "imagenet_eval", "ckpt_prediction"
    ]:

      eval_steps = model_params["num_eval_images"] // model_params["batch_size"]
      tf.logging.info("start computing eval metrics...")
      classifier = tf.estimator.Estimator(
          model_fn=model_fn_w_pruning, params=model_params)
      evaluation_metrics = classifier.evaluate(
          input_fn=data_input.input_fn,
          steps=eval_steps,
          checkpoint_path=ckpt_directory)
      tf.logging.info("finished per class accuracy eval.")
      return evaluation_metrics

  else:
    model_params["pruning_dict"] = pruning_params
    run_config = tf.estimator.RunConfig(
        save_summary_steps=300,
        save_checkpoints_steps=1000,
        log_step_count_steps=100)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn_w_pruning, config=run_config, params=model_params)
    tf.logging.info("start training...")
    classifier.train(
        input_fn=data_input.input_fn, max_steps=model_params["num_train_steps"])
    tf.logging.info("finished training.")
