# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Image embedding input and model functions for serving/inference."""

import tensorflow as tf

from official.vision.modeling import factory
from official.vision.ops import preprocess_ops
from official.vision.serving import export_base


MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class ImageEmbeddingModule(export_base.ExportModule):
  """Image Embedding Module."""

  def _build_model(self):
    input_specs = tf.keras.layers.InputSpec(
        shape=[self._batch_size] + self._input_image_size + [3])

    return factory.build_classification_model(
        input_specs=input_specs,
        model_config=self.params.task.model,
        l2_regularizer=None,
        skip_logits_layer=True)

  def _build_inputs(self, image):
    """Builds embedding model inputs for serving."""
    image = tf.image.resize(
        image, self._input_image_size, method=tf.image.ResizeMethod.BILINEAR)

    image = tf.reshape(
        image, [self._input_image_size[0], self._input_image_size[1], 3])

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)
    return image

  def serve(self, images):
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
    Returns:
      Tensor holding the normalized embedding.
    """
    with tf.device('cpu:0'):
      images = tf.cast(images, dtype=tf.float32)

      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._build_inputs,
              elems=images,
              fn_output_signature=tf.TensorSpec(
                  shape=self._input_image_size + [3], dtype=tf.float32),
              parallel_iterations=32))

    embedding = self.inference_step(images)
    embedding_norm = tf.nn.l2_normalize(embedding)

    return {'embedding': embedding, 'embedding_norm': embedding_norm}
