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

"""Example script for running VILA model."""

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
from lingvo import compat as tf
from lingvo.core import tokenizers as lingvo_tokenizers
from paxml import checkpoints
from paxml import learners
from paxml import tasks_lib
from paxml import train_states
from praxis import base_layer
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import schedules

from vila import coca_vila
from vila import coca_vila_configs


NestedMap = py_utils.NestedMap

_CKPT_DIR = flags.DEFINE_string('ckpt_dir', '', 'Path to checkpoint.')
_SPM_MODEL_PATH = flags.DEFINE_string(
    'spm_model_path', '', 'Path to sentence piece tokenizer model.'
)
_IMAGE_PATH = flags.DEFINE_string('image_path', '', 'Path to input image.')

_PRE_CROP_SIZE = 272
_IMAGE_SIZE = 224
_MAX_TEXT_LEN = 64
_TEXT_VOCAB_SIZE = 64000

_ZSL_QUALITY_PROMPTS = [
    ['good image', 'bad image'],
    ['good lighting', 'bad lighting'],
    ['good content', 'bad content'],
    ['good background', 'bad background'],
    ['good foreground', 'bad foreground'],
    ['good composition', 'bad composition'],
]


def load_vila_model(
    ckpt_dir,
):
  """Loads the VILA model from checkpoint directory.

  Args:
    ckpt_dir: The path to checkpoint directory

  Returns:
    VILA model, VILA model states
  """
  coca_config = coca_vila_configs.CocaVilaConfig()
  coca_config.model_type = coca_vila.CoCaVilaRankBasedFinetune
  coca_config.decoding_max_len = _MAX_TEXT_LEN
  coca_config.text_vocab_size = _TEXT_VOCAB_SIZE
  model_p = coca_vila_configs.build_coca_vila_model(coca_config)
  model_p.model_dims = coca_config.model_dims
  model = model_p.Instantiate()

  dummy_batch_size = 4  # For initialization only
  text_shape = (dummy_batch_size, 1, _MAX_TEXT_LEN)
  image_shape = (dummy_batch_size, _IMAGE_SIZE, _IMAGE_SIZE, 3)
  input_specs = NestedMap(
      ids=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.int32),
      image=jax.ShapeDtypeStruct(shape=image_shape, dtype=jnp.float32),
      paddings=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.float32),
      # For initialization only
      labels=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.float32),
      regression_labels=jax.ShapeDtypeStruct(
          shape=(dummy_batch_size, 10), dtype=jnp.float32
      ),
  )
  prng_key = jax.random.PRNGKey(123)
  prng_key, _ = jax.random.split(prng_key)
  vars_weight_params = model.abstract_init_with_metadata(input_specs)

  # `learner` is only used for initialization.
  learner_p = pax_fiddle.Config(learners.Learner)
  learner_p.name = 'learner'
  learner_p.optimizer = pax_fiddle.Config(
      optimizers.ShardedAdafactor,
      decay_method='adam',
      lr_schedule=pax_fiddle.Config(schedules.Constant),
  )
  learner = learner_p.Instantiate()

  train_state_global_shapes = tasks_lib.create_state_unpadded_shapes(
      vars_weight_params, discard_opt_states=False, learners=[learner]
  )
  model_states = checkpoints.restore_checkpoint(
      train_state_global_shapes, ckpt_dir
  )
  return model, model_states


def preprocess_image(
    image_path, pre_crop_size, image_size
):
  """Image preprocessing."""
  with tf.compat.v1.gfile.FastGFile(image_path, 'rb') as f:
    image_bytes = f.read()
  image = tf.io.decode_image(image_bytes, 3, expand_animations=False)
  image = tf.image.resize_bilinear(
      tf.expand_dims(image, 0), [pre_crop_size, pre_crop_size]
  )
  image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
  image = tf.cast(image, tf.float32)
  image = image / 255.0
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image.numpy()


def main(_):
  # Suppresses verbose INFO/DEBUG log.
  logging.set_verbosity(logging.ERROR)
  model, model_states = load_vila_model(_CKPT_DIR.value)
  image = preprocess_image(_IMAGE_PATH.value, _PRE_CROP_SIZE, _IMAGE_SIZE)
  input_batch = NestedMap(
      image=image,
      ids=jnp.zeros((1, 1, _MAX_TEXT_LEN), dtype=jnp.int32),
      paddings=jnp.zeros((1, 1, _MAX_TEXT_LEN), dtype=jnp.int32),
  )

  context_p = base_layer.JaxContext.HParams(do_eval=True)
  with base_layer.JaxContext(context_p):
    predictions = model.apply(
        {'params': model_states.mdl_vars['params']},
        input_batch,
        method=model.compute_predictions,
    )
    quality_scores = predictions['quality_scores']
  print('===== VILA predicted quality score [0, 1]: ', quality_scores)

  if _SPM_MODEL_PATH.value:
    tokenizer_p = lingvo_tokenizers.SentencePieceTokenizer.Params().Set(
        spm_model=_SPM_MODEL_PATH.value,
        vocab_size=_TEXT_VOCAB_SIZE,
    )
    tokenizer = tokenizer_p.Instantiate()

    all_prompts = [p for pair in _ZSL_QUALITY_PROMPTS for p in pair]  # pylint: disable=g-complex-comprehension
    ids, _, paddings = tokenizer.StringsToIds(all_prompts, max_length=4)
    context_p = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_p):
      input_batch = NestedMap(
          ids=ids.numpy(),
          paddings=paddings.numpy(),
          image=jnp.zeros((1, 224, 224, 3)),
      )
      text_encoded = model.apply(
          {'params': model_states.mdl_vars['params']},
          input_batch,
          method=model.compute_text_embedding,
      )
      text_embed = text_encoded.contrastive_txt_embed_norm
      image_embed = predictions.contrastive_img_embed_norm

      zsl_scores = jnp.matmul(image_embed, text_embed.T)

      zsl_scores = zsl_scores.reshape([-1, len(_ZSL_QUALITY_PROMPTS), 2])

      zsl_scores = jax.nn.softmax(zsl_scores)
      zsl_scores = zsl_scores.mean(1)
      zsl_scores = zsl_scores[:, 0]
      print('===== VILA ZSL predicted quality score [0, 1]: ', zsl_scores)


if __name__ == '__main__':
  app.run(main)
