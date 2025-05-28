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

"""Generate canary dataset.

Use the tensorflow_privacy library to generate secrets as per a configuration
and save them as tfrecord files.
"""

import configparser
import json
import os
import string
from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_privacy.privacy.privacy_tests.secret_sharer.generate_secrets import generate_text_secrets_and_references
from tensorflow_privacy.privacy.privacy_tests.secret_sharer.generate_secrets import SecretConfig
from tensorflow_privacy.privacy.privacy_tests.secret_sharer.generate_secrets import TextSecretProperties

from learn_to_forget.transformers import constants

CONFIG_FILE = flags.DEFINE_string('config_file', 'canary_config.ini',
                                  'Path to the canary config file')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config = configparser.ConfigParser()
  config.read(CONFIG_FILE.value)

  if not config.sections():
    raise app.UsageError('Empty config file.')

  secret_lengths = []
  secret_configs = []
  vocab = list(string.ascii_letters + string.digits)

  # Each section in the config corresponds to a pattern
  for canary_name in config.sections():
    num_repetitions = json.loads(config.get(canary_name, 'num_repetitions'))
    num_secrets = json.loads(
        config.get(canary_name, 'num_secrets_for_repetitions')
    )
    secret_length = config.getint(canary_name, 'secret_length')
    pattern = '{}' * secret_length
    secret_lengths.append(secret_length)

    secret_configs.append(
        SecretConfig(
            name=canary_name,
            num_repetitions=num_repetitions,
            num_secrets_for_repetitions=num_secrets,
            num_references=config.getint(canary_name, 'num_references'),
            properties=TextSecretProperties(vocab, pattern),
        ),
    )

  secret_sets = generate_text_secrets_and_references(secret_configs)

  os.makedirs(constants.CANARY_TFDS_PATH, exist_ok=True)

  # Create tf.data.Dataset from secrets
  features = tfds.features.FeaturesDict({
      'inputs': tfds.features.Text(),
      'targets': tfds.features.Text(),
      'num_repetitions': tf.int64,
  })

  # Each secret is a SecretsSet corresponding to the secret type
  for secret_set, secret_length in zip(secret_sets, secret_lengths):
    secret_set_dict = secret_set.secrets
    references = secret_set.references

    for num_repetitions, secrets in secret_set_dict.items():
      with tf.io.TFRecordWriter(
          constants.CANARY_TFDS_DATA_PATTERN.format(
              secret_length, num_repetitions
          )
      ) as writer:
        # Repeat the secrets as per the config
        for _ in range(num_repetitions):
          for secret in secrets:
            data = {
                'inputs': 'My secret is ',
                'targets': secret,
                'num_repetitions': num_repetitions
            }
            secret_bytes = features.serialize_example(data)
            writer.write(secret_bytes)

      # Writing the references for this secret set
      with tf.io.TFRecordWriter(
          constants.CANARY_TFDS_TEST_DATA_PATTERN.format(
              secret_length, num_repetitions
          )
      ) as writer:
        for reference in references:
          data = {
              'inputs': 'My secret is ',
              'targets': reference,
              'num_repetitions': num_repetitions
          }
          reference_bytes = features.serialize_example(data)
          writer.write(reference_bytes)

if __name__ == '__main__':
  app.run(main)
