# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
# pylint: disable=line-too-long
r"""Run inference on random sequences many times, saving the top results.

This module is designed to run inference on a trained aptamers TF model
to score many sequences and then write out the top scoring sequences to
be used in other analysis. For example, these top predicted sequences
can read in the colab picking sequences in order
to validate the TF model.

This module uses the standard aptamers tensorflow infrastructure to run
the inference. This is the inferer class defined in
learning/eval_feedforward and described in the Aptamers TensorFlow Doc.

Inference is generally run in batches of sequences, so a MapFlat function
is used to combine the results of interference into a flattened list
of (sequence, score) pairs. (In other words, each map function returns
a list of (sequence, score) pairs and these results are combined into
one list of pairs. Using a standard Map function would result in a
list of lists which is not what we want.) From there, the Top.Of
combiner returns only the top results, sorted by highest score. These
are then written to a text file.

For now, each batch creates its own inferer, causing each map function
to have more overhead. Because of this,batch sizes around the default of
10000 are performant. Fixing this in the future means converting the
MapFlat to a ParDo and initializing the inferer in start_bundle.
This module runs inference 1 billion times in 50 minutes so it
is sufficient for now

Run locally with:

  :search_inference -- --flume_exec_mode=UNOPT \
  --target_name='target' \
  --model_dir=xxx \
   --checkpoint_path='model.ckpt-312500' \
   --output_name=xxx/base30_1B_inference_top2k

"""
# pylint: enable=line-too-long

# Google internal
import apache_beam as beam
import runner
import app
import flags


from ..learning import eval_feedforward
from ..utils import pool

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_batches', 1000, 'Number of batches to run')
flags.DEFINE_integer('batch_size', 10000, 'Number of sequences per batch')
flags.DEFINE_integer('num_to_save', 2000, 'The number of top results to save.')
flags.DEFINE_string('target_name', None,
                    'The name of the target protein for the inference.')
flags.DEFINE_string('model_dir', None,
                    'The path to base trained model directory.')
flags.DEFINE_string('checkpoint_path', None,
                    'String path to the checkpoint of the model.')
flags.DEFINE_string('output_name', None, 'The name of the output file.')
flags.DEFINE_integer('sequence_length', 40, 'The length of sequences to test.')
flags.DEFINE_string('affinity_target_map', None, 'Name of affinity target map')


_METRICS_NAMESPACE = 'SearchInference'


class RunInferenceForBatch(beam.PTransform):
  """Generates random batches and runs inference."""

  def __init__(self, sequence_length, target_name, model_dir, checkpoint_path,
               affinity_target_map):
    self._sequence_length = sequence_length
    self._target_name = target_name
    self._model_dir = model_dir
    self._checkpoint_path = checkpoint_path
    self._affinity_target_map = affinity_target_map
    self._inferer = None  # Lazy initialization

  def _run_inference_for_batch(self, batch_size):
    beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'num_batches').inc()
    if not self._inferer:
      self._inferer = eval_feedforward.create_inferer(self._model_dir,
                                                      self._checkpoint_path,
                                                      self._affinity_target_map)
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'loaded_inferer').inc()

    seqs = pool.random_sequences(
        batch_size, sequence_length=self._sequence_length)
    scores = self._inferer.get_affinities_for_sequences(seqs, self._target_name)
    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 'num_inference').inc(len(scores))
    return list(zip(seqs, scores))

  def expand(self, p_coll):
    return p_coll | beam.FlatMap(self._run_inference_for_batch)


def main(argv=()):
  del argv  # Unused.

  def pipeline(root):
    # pylint: disable=expression-not-assigned
    (root
     | 'Range' >> beam.Create(
         [FLAGS.batch_size for _ in range(FLAGS.num_batches)])
     | RunInferenceForBatch(FLAGS.sequence_length, FLAGS.target_name,
                            FLAGS.model_dir, FLAGS.checkpoint_path,
                            FLAGS.affinity_target_map)
     | 'TopByValue' >> beam.transforms.combiners.Top.Of(
         FLAGS.num_to_save, compare=lambda a, b: (a[1], a[0]) < (b[1], b[0]))
     | 'save' >> beam.io.WriteToText(FLAGS.output_name))
    # pylint: enable=expression-not-assigned

  runner.FlumeRunner().run(pipeline)


if __name__ == '__main__':
  app.run()
