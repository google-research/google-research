# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Contains callbacks for writing tf.summary for losses and point cloud visualization and saving checkpoint.
"""
import logging
import os

import gin
import gin.tf
import tensorflow as tf
from tf3d import standard_fields
from tf3d.utils import box_utils
from tf3d.utils import colormap
from tensorboard.plugins.mesh import summary_v2 as mesh_summary


@gin.configurable
class CustomTensorBoard(tf.keras.callbacks.Callback):
  """Custom TensorBoard logging class for losses and point cloud visualization."""

  def __init__(
      self,
      log_dir='logs',
      num_qualitative_examples=-1,
      metric_classes=None,
      split='val',
      batch_update_freq=100,
      num_steps_per_epoch=1000,
      visualizer_field_of_view=75,
      point_size=.01,
      max_num_points_qualitative=100000,
  ):
    """Custom TensorBoard logging class for losses and point cloud visualization.

    Args:
      log_dir: The directory where subfolder for logs are created.
      num_qualitative_examples: The number of examples to save for qualitative
        visualization.
      metric_classes: A list of metric classes.
      split: The prefix str for subfolders of eval logs.
      batch_update_freq: An int of number of steps between logging during
        training.
      num_steps_per_epoch: A int used for recovering the current eval steps.
      visualizer_field_of_view: An angle of the field of view for point cloud
        visualization.
      point_size: A float of point size for point cloud visualization.
      max_num_points_qualitative: Maximum number of points for visualization.
    """
    self.log_dir = log_dir
    self._writers = {}
    self._batch_update_freq = batch_update_freq
    self._global_train_batch = 0
    self._global_val_batch = 0
    self.num_qualitative_examples = num_qualitative_examples
    self.split = split
    self.max_num_points_qualitative = max_num_points_qualitative

    # for calculating actual number of steps.
    self.num_steps_per_epoch = num_steps_per_epoch
    self.epoch_number = 1

    self._mesh_config_dict = {
        'camera': {
            'cls': 'PerspectiveCamera',
            'fov': visualizer_field_of_view
        },
        'material': {
            'cls': 'PointsMaterial',
            'size': point_size
        },
    }
    self._pascal_color_map = None
    self._metric_classes = metric_classes

  @property
  def _train_writer(self):
    """A lazily initialized summary writer for training."""
    if 'train' not in self._writers:
      self._writers['train'] = tf.summary.create_file_writer(
          self._train_dir)
    return self._writers['train']

  @property
  def _val_writer(self):
    """A lazily initialized metric summary writer for validation."""
    if 'val' not in self._writers:
      self._writers['val'] = tf.summary.create_file_writer(
          self._val_dir)
    return self._writers['val']

  @property
  def _val_mesh_writer(self):
    """A lazily initialized mesh summary writer for validation."""
    if 'val_mesh' not in self._writers:
      self._writers['val_mesh'] = tf.summary.create_file_writer(
          self._val_mesh_dir)
    return self._writers['val_mesh']

  def _close_and_clear_mesh_writer(self):
    if 'val_mesh' in self._writers:
      self._writers['val_mesh'].close()
      del self._writers['val_mesh']

  def _clear_mesh_event_files(self):
    old_events = tf.io.gfile.glob(os.path.join(self._val_mesh_dir, 'event*'))
    if old_events:
      for event_file in old_events:
        logging.info('removing old mesh event file:%s', event_file)
        tf.io.gfile.remove(event_file)

  @property
  def _metric(self):
    """A lazily initialized metric object for validation."""
    if hasattr(self, '_metric_singleton'):
      return self._metric_singleton
    if self._metric_classes and hasattr(self, 'model'):
      self._metric_singleton = []
      for metric_class in self._metric_classes:
        self._metric_singleton.append(
            metric_class(eval_prefix='eval_' + self.split))
      return self._metric_singleton
    else:
      return None

  def set_model(self, model):
    """Sets Keras model and step counter."""
    self.model = model

    self._train_dir = os.path.join(self.log_dir, 'train')
    self._train_step = self.model._train_counter  # pylint: disable=protected-access

    self._val_dir = os.path.join(self.log_dir, 'eval_' + self.split)
    self._val_mesh_dir = os.path.join(self.log_dir,
                                      'eval_' + self.split + '_mesh')
    self._val_step = self.model._predict_counter  # pylint: disable=protected-access
    logging.info('Calling set model at val step: %d', self._val_step)

  def set_epoch_number(self, epoch):
    self.epoch_number = epoch

  def on_train_begin(self, logs=None):
    self._global_train_batch = 0

  def on_train_end(self, logs=None):
    for writer in self._writers.values():
      writer.close()

  def on_epoch_begin(self, epoch, logs=None):
    pass

  def on_epoch_end(self, epoch, logs=None):
    pass

  def on_train_batch_begin(self, batch, logs=None):
    self._global_train_batch += 1

  def on_train_batch_end(self, batch, logs=None):
    """Optionally write scalar summaries of losses and learning rate at the end of each training batch."""
    with self._train_writer.as_default():
      with tf.summary.record_if(batch % self._batch_update_freq == 0):
        tf.summary.scalar(
            'learning_rate',
            self.model.optimizer.learning_rate(self._train_step),
            step=self._train_step)
        for loss_name, loss_value in self.model.loss_names_to_losses.items():
          tf.summary.scalar(loss_name, loss_value, step=self._train_step)

  def on_predict_begin(self, logs=None):
    self._global_val_batch = 0
    self._pascal_color_map = colormap.create_pascal_color_map()
    if self._metric:
      for metric in self._metric:
        metric.reset_states()
    self._clear_mesh_event_files()

  def on_predict_end(self, logs=None):
    if self._metric:
      step = self.epoch_number * self.num_steps_per_epoch
      with self._val_writer.as_default():
        for metric in self._metric:
          metric_dict = metric.get_metric_dictionary()
          for metric_name, value in metric_dict.items():
            tf.summary.scalar(name=metric_name, data=value, step=step)
            logging.info('at step %d - %s: %f', step, metric_name, value)
      self._val_writer.flush()
    self._val_mesh_writer.flush()
    self._close_and_clear_mesh_writer()

  def on_predict_batch_begin(self, batch, logs=None):
    self._global_val_batch += 1

  def on_predict_batch_end(self, batch, logs=None):
    """Write mesh summaries of semantics groundtruth and prediction point clouds at the end of each validation batch."""
    inputs = logs['inputs']
    outputs = logs['outputs']
    if self._metric:
      for metric in self._metric:
        metric.update_state(inputs=inputs, outputs=outputs)

    if batch <= self.num_qualitative_examples:
      # point cloud visualization
      vertices = tf.reshape(
          inputs[standard_fields.InputDataFields.point_positions], [-1, 3])
      num_valid_points = tf.squeeze(
          inputs[standard_fields.InputDataFields.num_valid_points])
      logits = outputs[
          standard_fields.DetectionResultFields.object_semantic_points]
      num_classes = logits.get_shape().as_list()[-1]
      logits = tf.reshape(logits, [-1, num_classes])
      gt_semantic_class = tf.reshape(
          inputs[standard_fields.InputDataFields.object_class_points], [-1])

      vertices = vertices[:num_valid_points, :]
      logits = logits[:num_valid_points, :]
      gt_semantic_class = gt_semantic_class[:num_valid_points]
      max_num_points = tf.math.minimum(self.max_num_points_qualitative,
                                       num_valid_points)
      sample_indices = tf.random.shuffle(
          tf.range(num_valid_points))[:max_num_points]
      vertices = tf.gather(vertices, sample_indices)
      logits = tf.gather(logits, sample_indices)
      gt_semantic_class = tf.gather(gt_semantic_class, sample_indices)
      semantic_class = tf.math.argmax(logits, axis=1)
      pred_colors = tf.gather(self._pascal_color_map, semantic_class, axis=0)
      gt_colors = tf.gather(self._pascal_color_map, gt_semantic_class, axis=0)

      if standard_fields.InputDataFields.point_colors in inputs:
        point_colors = (tf.reshape(
            inputs[standard_fields.InputDataFields.point_colors], [-1, 3]) +
                        1.0) * 255.0 / 2.0
        point_colors = point_colors[:num_valid_points, :]
        point_colors = tf.gather(point_colors, sample_indices)
        point_colors = tf.math.minimum(point_colors, 255.0)
        point_colors = tf.math.maximum(point_colors, 0.0)
        point_colors = tf.cast(point_colors, dtype=tf.uint8)
      else:
        point_colors = tf.ones_like(vertices, dtype=tf.uint8) * 128

      # add points and colors for predicted objects
      if standard_fields.DetectionResultFields.objects_length in outputs:
        box_corners = box_utils.get_box_corners_3d(
            boxes_length=outputs[
                standard_fields.DetectionResultFields.objects_length],
            boxes_height=outputs[
                standard_fields.DetectionResultFields.objects_height],
            boxes_width=outputs[
                standard_fields.DetectionResultFields.objects_width],
            boxes_rotation_matrix=outputs[
                standard_fields.DetectionResultFields.objects_rotation_matrix],
            boxes_center=outputs[
                standard_fields.DetectionResultFields.objects_center])
        box_points = box_utils.get_box_as_dotted_lines(box_corners)

        objects_class = tf.reshape(
            outputs[standard_fields.DetectionResultFields.objects_class], [-1])
        box_colors = tf.gather(self._pascal_color_map, objects_class, axis=0)
        box_colors = tf.repeat(
            box_colors[:, tf.newaxis, :], box_points.shape[1], axis=1)
        box_points = tf.reshape(box_points, [-1, 3])
        box_colors = tf.reshape(box_colors, [-1, 3])
        pred_vertices = tf.concat([vertices, box_points], axis=0)
        pred_colors = tf.concat([pred_colors, box_colors], axis=0)
      else:
        pred_vertices = vertices

      # add points and colors for gt objects
      if standard_fields.InputDataFields.objects_length in inputs:
        box_corners = box_utils.get_box_corners_3d(
            boxes_length=tf.reshape(
                inputs[standard_fields.InputDataFields.objects_length],
                [-1, 1]),
            boxes_height=tf.reshape(
                inputs[standard_fields.InputDataFields.objects_height],
                [-1, 1]),
            boxes_width=tf.reshape(
                inputs[standard_fields.InputDataFields.objects_width], [-1, 1]),
            boxes_rotation_matrix=tf.reshape(
                inputs[standard_fields.InputDataFields.objects_rotation_matrix],
                [-1, 3, 3]),
            boxes_center=tf.reshape(
                inputs[standard_fields.InputDataFields.objects_center],
                [-1, 3]))
        box_points = box_utils.get_box_as_dotted_lines(box_corners)

        objects_class = tf.reshape(
            inputs[standard_fields.InputDataFields.objects_class], [-1])
        box_colors = tf.gather(self._pascal_color_map, objects_class, axis=0)
        box_colors = tf.repeat(
            box_colors[:, tf.newaxis, :], box_points.shape[1], axis=1)

        box_points = tf.reshape(box_points, [-1, 3])
        box_colors = tf.reshape(box_colors, [-1, 3])
        gt_vertices = tf.concat([vertices, box_points], axis=0)
        gt_colors = tf.concat([gt_colors, box_colors], axis=0)
      else:
        gt_vertices = vertices
      if batch == 1:
        logging.info('writing point cloud(shape %s) to summery.',
                     gt_vertices.shape)
      if standard_fields.InputDataFields.camera_image_name in inputs:
        camera_image_name = str(inputs[
            standard_fields.InputDataFields.camera_image_name].numpy()[0])
      else:
        camera_image_name = str(batch)
      logging.info(camera_image_name)
      with self._val_mesh_writer.as_default():
        mesh_summary.mesh(
            name=(self.split + '_points/' + camera_image_name),
            vertices=tf.expand_dims(vertices, axis=0),
            faces=None,
            colors=tf.expand_dims(point_colors, axis=0),
            config_dict=self._mesh_config_dict,
            step=self._val_step,
        )
        mesh_summary.mesh(
            name=(self.split + '_predictions/' + camera_image_name),
            vertices=tf.expand_dims(pred_vertices, axis=0),
            faces=None,
            colors=tf.expand_dims(pred_colors, axis=0),
            config_dict=self._mesh_config_dict,
            step=self._val_step,
        )
        mesh_summary.mesh(
            name=(self.split + '_ground_truth/' + camera_image_name),
            vertices=tf.expand_dims(gt_vertices, axis=0),
            faces=None,
            colors=tf.expand_dims(gt_colors, axis=0),
            config_dict=self._mesh_config_dict,
            step=self._val_step,
        )
      if batch == self.num_qualitative_examples:
        self._val_mesh_writer.flush()


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
  """Custom checkpoint saving class for saving model weights and epoch."""

  def __init__(self, ckpt_dir, save_epoch_freq=1, max_to_keep=3):
    self._ckpt_saved_epoch = tf.Variable(
        initial_value=tf.constant(
            -1, dtype=tf.dtypes.int64),
        name='ckpt_saved_epoch')
    self.ckpt_dir = ckpt_dir
    self.max_to_keep = max_to_keep
    self.save_epoch_freq = save_epoch_freq

  def set_model(self, model):
    self._model = model
    checkpoint = tf.train.Checkpoint(
        model=self._model, ckpt_saved_epoch=self._ckpt_saved_epoch)
    self.write_checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=self.ckpt_dir, max_to_keep=self.max_to_keep)

  def _save(self, epoch):
    self._ckpt_saved_epoch.assign(epoch)
    self.write_checkpoint_manager.save(epoch)
    logging.info('Saving ckpt for epoch: %d at %s', epoch, self.ckpt_dir)

  def on_epoch_begin(self, epoch, logs=None):
    if epoch == 0:
      self._save(epoch)

  def on_epoch_end(self, epoch, logs=None):
    if epoch % self.save_epoch_freq == 0:
      save_epoch = epoch + 1
      self._save(save_epoch)
