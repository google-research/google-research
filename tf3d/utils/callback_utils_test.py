# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for tf3d.utils.callback_utils."""

import os
import tensorflow as tf

from tf3d import standard_fields
from tf3d.utils import callback_utils


class CallbackUtilsTest(tf.test.TestCase):

  def test_custom_tensorboard(self):
    log_dir = '/tmp/tf3d/callback_util_test'
    if tf.io.gfile.exists(log_dir):
      tf.io.gfile.rmtree(log_dir)

    callback = callback_utils.CustomTensorBoard(
        log_dir=log_dir,
        metric_classes=None,
        batch_update_freq=1,
        num_qualitative_examples=10,
        split='val')
    model = tf.keras.Model()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                0.01, 1, 0.1)))
    model.loss_names_to_losses = {'total_loss': 5.}
    inputs = {
        standard_fields.InputDataFields.point_positions:
            tf.constant([[[3., 2., 1.], [2., 3., 1.]]]),
        standard_fields.InputDataFields.num_valid_points:
            tf.constant([1]),
        standard_fields.InputDataFields.object_class_points:
            tf.constant([[[0], [1]]]),
        # gt objects
        standard_fields.InputDataFields.objects_length:
            tf.constant([[[3.]]]),
        standard_fields.InputDataFields.objects_height:
            tf.constant([[[1.]]]),
        standard_fields.InputDataFields.objects_width:
            tf.constant([[[2.]]]),
        standard_fields.InputDataFields.objects_center:
            tf.constant([[[0., 0., 0.]]]),
        standard_fields.InputDataFields.objects_rotation_matrix:
            tf.eye(3, 3)[tf.newaxis, tf.newaxis, Ellipsis],
        standard_fields.InputDataFields.objects_class:
            tf.constant([[[1]]]),
        standard_fields.InputDataFields.camera_image_name:
            tf.convert_to_tensor([['image1', 'image2']], dtype=tf.string)
    }
    outputs = {
        standard_fields.DetectionResultFields.object_semantic_points:
            tf.constant([[[3., 2.], [2., 3.]]]),
        # predicted objects
        standard_fields.DetectionResultFields.objects_length:
            tf.constant([[3.]]),
        standard_fields.DetectionResultFields.objects_height:
            tf.constant([[1.]]),
        standard_fields.DetectionResultFields.objects_width:
            tf.constant([[2.]]),
        standard_fields.DetectionResultFields.objects_center:
            tf.constant([[0., 0., 0.]]),
        standard_fields.DetectionResultFields.objects_rotation_matrix:
            tf.expand_dims(tf.eye(3, 3), axis=0),
        standard_fields.DetectionResultFields.objects_class:
            tf.constant([[1]]),
    }

    callback.set_model(model)
    callback.on_train_begin()
    callback.on_epoch_begin(epoch=1, logs=None)
    callback.on_train_batch_begin(batch=1, logs=None)
    callback.on_train_batch_end(batch=1, logs=None)
    callback.on_epoch_end(epoch=1, logs=None)
    callback.on_train_end()
    self.assertNotEmpty(
        (tf.io.gfile.glob(os.path.join(log_dir, 'train/events*'))))

    callback.on_predict_begin()
    callback.on_predict_batch_begin(batch=1, logs=None)
    callback.on_predict_batch_end(
        batch=1, logs={
            'outputs': outputs,
            'inputs': inputs
        })
    callback.on_predict_end()
    self.assertEmpty(
        (tf.io.gfile.glob(os.path.join(log_dir, 'eval_val/events*'))))
    self.assertNotEmpty(
        (tf.io.gfile.glob(os.path.join(log_dir, 'eval_val_mesh/events*'))))

  def test_custom_model_checkpoint(self):
    ckpt_dir = '/tmp/tf3d/callback_util_test'
    if tf.io.gfile.exists(ckpt_dir):
      tf.io.gfile.rmtree(ckpt_dir)

    callback = callback_utils.CustomModelCheckpoint(ckpt_dir=ckpt_dir,
                                                    save_epoch_freq=1,
                                                    max_to_keep=5)
    model = tf.keras.Model()
    callback.set_model(model)
    callback.on_epoch_begin(epoch=0, logs=None)
    callback.on_epoch_end(epoch=1, logs=None)
    self.assertNotEmpty(
        (tf.io.gfile.glob(os.path.join(ckpt_dir, '*'))))


if __name__ == '__main__':
  tf.test.main()
