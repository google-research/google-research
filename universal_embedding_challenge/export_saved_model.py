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

r"""Vision models export binary for serving/inference.

To export a trained checkpoint in saved_model format (shell script):

EXPERIMENT_TYPE = XX
CHECKPOINT_PATH = XX
EXPORT_DIR_PATH = XX
export_saved_model --experiment=${EXPERIMENT_TYPE} \
                   --export_dir=${EXPORT_DIR_PATH}/ \
                   --checkpoint_path=${CHECKPOINT_PATH} \
                   --batch_size=1

To serve (python):

image_tensor = XX
imported = tf.saved_model.load(export_dir_path)
model_fn = imported.signatures['serving_default']
embedding = model_fn(image_tensor)['embedding_norm']
"""

from absl import app
from absl import flags

# pylint: disable=unused-import
from official.vision.modeling.backbones import vit  # pylint: disable=g-bad-import-order
from universal_embedding_challenge import image_classification
# pylint: enable=unused-import

from universal_embedding_challenge import image_embedding
from official.core import exp_factory
from official.vision.serving import export_saved_model_lib


FLAGS = flags.FLAGS

_EXPERIMENT = flags.DEFINE_string(
    'experiment', None,
    'experiment type, e.g. vit_with_bottleneck_imagenet_pretrain')
_EXPORT_DIR = flags.DEFINE_string('export_dir', None, 'The export directory.')
_CHECKPOINT_PATH = flags.DEFINE_string('checkpoint_path', None,
                                       'Checkpoint path.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 1, 'The batch size.')
_RESOLUTION = flags.DEFINE_integer('resolution', 224,
                                   'The resolution of the image.')


def main(_):

  params = exp_factory.get_exp_config(_EXPERIMENT.value)
  params.validate()
  params.lock()

  export_module = image_embedding.ImageEmbeddingModule(
      params=params,
      batch_size=_BATCH_SIZE.value,
      input_image_size=[_RESOLUTION.value, _RESOLUTION.value],
      input_type='image_tensor',
      num_channels=3)

  export_saved_model_lib.export_inference_graph(
      input_type='image_tensor',
      batch_size=_BATCH_SIZE.value,
      input_image_size=[_RESOLUTION.value, _RESOLUTION.value],
      params=params,
      export_module=export_module,
      checkpoint_path=_CHECKPOINT_PATH.value,
      export_dir=_EXPORT_DIR.value,
      export_checkpoint_subdir='checkpoint',
      export_saved_model_subdir='saved_model',
      log_model_flops_and_params=False)


if __name__ == '__main__':
  app.run(main)
