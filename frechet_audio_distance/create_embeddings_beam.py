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

"""A Python Beam pipeline that extracts embeddings form audio tf.examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import ReadFromTFRecord
from apache_beam.io import WriteToTFRecord
from apache_beam.transforms import window
from apache_beam.utils import windowed_value

import numpy as np
import scipy.io.wavfile
import scipy.spatial.distance
import tensorflow.compat.v1 as tf

from frechet_audio_distance.audioset_model import AudioSetModel


def _int64_feature(value):
  """Helper function for creating an int64 tf.Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Helper function for creating an float tf.Feature."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class ComputeMeanAndCovariance(beam.CombineFn):
  """Accumulates and computes the mean and convariance matrix.

   The covariance matrix sigma is computed on mean normalized data X like this:
   m = np.mean(X, axis=0) # mean of X = (x_0, ..., x_i, ... x_n)
     = sum(x_i) / n => sum(x_i) = m * n
   sigma = sum[(x_i - m) * (x_i - m).T] / (n-1)
         = sum [ x_i * x_i.T - x_i * m.T - m * x_i.T -  m * m.T]/(n-1)
         = [sum(x_i * x_i.T) - sum(x_i) * m.T - m * sum(x_i)  + n m * m.T]/(n-1)
         = [sum(x_i * x_i.T) - n * m * m.T - m * n * m.T  + n m * m.T]/(n-1)
         = [sum(x_i * x_i.T) - n * m * m.T]/(n-1)
         = sum(x_i * x_i.T)/(n-1) - (m * m.T) * n/(n-1)
    This equivalent to: sigma = np.cov(X, rowvar=0) but runs much faster:
      - np.cov: 400 hour of audio 2048 dim ~ 1h
      - this approach: 400 hour of audio 2048 dim ~ 5min

    By splitting sigma up this way it can be accumulated in parallel by using
    accumulators for:
      1) sum(x_i * x_i.T)
      2) for the mean sum(x_i)
      3) and an accumulator that just counts the total number samples.

  The resulting PCollection contains a single tf.Example containing the stats.
  """

  def __init__(self, key_name, embedding_dim):
    """Initalizes ComputeMeanAndCovariance with name and embedding_dim.

    Args:
      key_name: Identifier for the set of examples processed in this pipeline.
      embedding_dim: Dimensionality of the embeddings.
    """
    self._key_name = key_name
    self._embedding_dim = embedding_dim

  def create_accumulator(self):
    """See base class."""
    mean_accu = np.zeros((self._embedding_dim, 1), dtype=np.float64)
    cov_accu = np.zeros((self._embedding_dim, self._embedding_dim),
                        dtype=np.float64)

    return mean_accu, cov_accu, 0

  def add_input(self, accu, element):
    """See base class."""
    mean_accu, cov_accu, sample_count = accu
    for embeddding in element:
      if self._embedding_dim != len(embeddding):
        raise ValueError('Embedding dims missmatch: %d != %d' %
                         (self._embedding_dim, len(embeddding)))
      np_embeddding = np.array(embeddding).reshape((self._embedding_dim, 1))
      mean_accu += np_embeddding
      cov_accu += np_embeddding * np_embeddding.T
      sample_count += 1
    return mean_accu, cov_accu, sample_count

  def merge_accumulators(self, accumulators):
    """See base class."""
    merged_mean, merged_cov, merged_sample_count = self.create_accumulator()
    for accu in accumulators:
      mean_accu, cov_accu, sample_count = accu
      merged_mean += mean_accu
      merged_cov += cov_accu
      merged_sample_count += sample_count
    return merged_mean, merged_cov, merged_sample_count

  def extract_output(self, accu):
    """See base class."""
    mean_accu, cov_accu, sample_count = accu
    feature = {
        'embedding_count': _int64_feature([sample_count]),
        'embedding_length': _int64_feature([self._embedding_dim])
    }
    if sample_count > 0:
      mu = (mean_accu / sample_count).reshape((self._embedding_dim, 1))
      sigma = cov_accu / (sample_count - 1) - mu * mu.T * (sample_count) / (
          sample_count - 1)
      feature['mu'] = _float_feature(list(mu.flatten()))
      feature['sigma'] = _float_feature(list(sigma.flatten()))
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return self._key_name, example


class BatchedInference(beam.DoFn):
  """Performs inference on batches of tf.Examples."""

  def __init__(self,
               batch_size,
               model,
               feature_key=None,
               distortion_fn=None,
               max_clip_samples=None,
               audio_output_name=None,
               compute_metrics=False):
    """Initializes BatchedInference.

    Args:
      batch_size: Number of examples to batch.
      model: ModelConfig namedtuple; contains model name, embedding dimension
        size and parameter configuration dictionary.
      feature_key: tf.example feature that contains the samples that are to be
        processed.
      distortion_fn: function that takes numpy vector of samples, distorts them
        and returns a a numpy vector of them same size.
      max_clip_samples: Each audio clip is truncated to this value if it's not
        set to 'None'.
      audio_output_name: When set the distorted audio is yielded as as a
        tf.train.Feature with this name.
      compute_metrics: When true then the other, non-fad metrics are computed
        for each distortion.
    """
    self._batch_size = batch_size
    self._buffer = []
    self._embedding_dim = model.embedding_dim
    self._step_size = model.step_size
    self._model_ckpt = model.model_ckpt
    self._model = None
    self._feature_key = feature_key or 'audio/reference/raw_audio'
    self._distortion_fn = distortion_fn or (lambda x: x)
    self._audio_output_name = audio_output_name
    self._max_clip_samples = max_clip_samples
    self._compute_metrics = compute_metrics

  def _floats(self, example):
    """Extracts the samples as a list of floats."""
    samples_np = example.features.feature[self._feature_key].float_list.value
    if self._max_clip_samples:
      samples_np = samples_np[:self._max_clip_samples]
    return samples_np

  def _window(self, output, add_window=False):
    """Forces an output into the global window.

    While 'process' will output to the same window as its incomming element,
    'finish_bundle' has to specify BatchedInferencea window to output into.
    Since we are dealing with a bounded input, we can use 'GlobalWindow'.

    Args:
      output: The function output that may need to be added to a window.
      add_window: Adds output to the GlobalWindow.

    Returns:
      output or output encapsulated in 'WindowedValue'.
    """
    if add_window:
      return windowed_value.WindowedValue(output, -1, [window.GlobalWindow()])
    return output

  def _get_metrics(self, clean_audio, noise_audio, dist_samples):
    """Add other metrics to the result."""
    cos_dis = scipy.spatial.distance.cosine(clean_audio, dist_samples)
    feature = {
        'cos': _float_feature([cos_dis]),
        'num_samples': _float_feature([float(dist_samples.shape[0])])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

  def _flush_buffer(self, add_window=False):
    """Calls the model with the examples in the buffer to create embeddings.

    Args:
      add_window: When true outputs are added to the GlobalWindow.

    Yields:
      - main output: Original key + TF.Example containing the embedding.
      - raw (side output): The raw embeddings from the model.
      - metrics (side output, optional): Original key + TF.Example.
      - distorted_audio (side output, optional):  Original key + TF.Example
          containing the distorted audio.

    Raises:
      ValueError:
        - When the extracted input features are not finite.
        - When the computed embeddings are not finite.
        - If the emddings do not have the expected dimension.
    """
    batch = []
    keys = []
    for key, example in self._buffer:
      clean_samples = np.array(self._floats(example))
      dist_samples = self._distortion_fn(clean_samples)
      if self._compute_metrics:
        noise_samples = dist_samples - clean_samples
        metrics = self._get_metrics(clean_samples, noise_samples, dist_samples)
        yield beam.pvalue.TaggedOutput('metrics',
                                       self._window((key, metrics), add_window))
      if self._audio_output_name:
        audio_output = (key, (self._audio_output_name,
                              _float_feature(dist_samples.tolist())))
        yield beam.pvalue.TaggedOutput('distorted_audio',
                                       self._window(audio_output, add_window))
      for features in self._model.extract_features(dist_samples):
        if not np.isfinite(features).all():
          raise ValueError('Input Feature not finite %s' % key)
        batch.append(features)
        keys.append(key)
    embeddings = self._model.process_batch(np.concatenate(batch))
    yield beam.pvalue.TaggedOutput(
        'raw', self._window(embeddings.tolist(), add_window))
    for key, embedding_vector in zip(keys, embeddings.tolist()):
      if self._embedding_dim != len(embedding_vector):
        raise ValueError('Embedding isn\'t the expected dimension %d vs %d' %
                         (len(embedding_vector), self._embedding_dim))
      if not np.isfinite(embedding_vector).all():
        raise ValueError('Embedding not finite %s' % key)
      feature = {
          'embedding': _float_feature(embedding_vector),
          'embedding_count': _int64_feature([1]),
          'embedding_length': _int64_feature([self._embedding_dim])
      }
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      yield self._window((key, example), add_window)
    del self._buffer[:]

  def start_bundle(self):
    """Initializes the model on the worker."""
    # Because AudioSetModel is not serializable it can't be initialized
    # in __init__  and has to be initialized here. A bundle consists of
    # a call to start_bundle followed by many calls to process
    # and then a final call to finish_bundle.
    self._model = AudioSetModel(self._model_ckpt, self._step_size)

  def process(self, element):
    """Buffers input; the model is more efficient when called in batch mode."""
    self._buffer.append(element)
    if len(self._buffer) == self._batch_size:
      for output in self._flush_buffer():
        yield output

  def finish_bundle(self):
    """Processes the final examples still in the buffer prior to termination."""
    if self._buffer:
      for output in self._flush_buffer(add_window=True):
        yield output


def create_audio_example(feature_key, samples, name):
  """Wraps samples in a tf.example with using the provided feature_key."""
  feature = {feature_key: _float_feature(samples)}
  feature['name'] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[name.encode('utf-8')]))
  return tf.train.Example(features=tf.train.Features(feature=feature))


class ReadWavFiles(beam.DoFn):
  """Read a wav file and wrap the data in a tf.example proto."""

  def process(self, element):
    """See base class."""
    _, data = scipy.io.wavfile.read(element)
    example = create_audio_example('audio/reference/raw_audio', data, element)
    yield element, example


class AddKey(beam.DoFn):
  """Add a key to value and create a key,value pair."""

  def process(self, element):
    """See base class."""
    if 'name' in element.features.feature:
      yield element.features.feature['name'], element
    else:
      yield str(uuid.uuid4()), element


class DropKey(beam.DoFn):
  """Drop the key from a key, value pair."""

  def process(self, element):
    """See base class."""
    _, value = element
    yield value


def create_pipeline(embedding_model,
                    files_input_list=None,
                    tfrecord_input=None,
                    embeddings_output=None,
                    stats_output=None,
                    feature_key=None,
                    name='all_train_embeddings',
                    batch_size=64):
  """Returns a pipeline that extracts stats from audio examples.

  Args:
    embedding_model: ModelConfig namedtuple; contains model ckpt, embedding
      dimension size and step size.
    files_input_list: List of files from where the audio is to be read.
    tfrecord_input: Path to a tfrecord containing audio.
    embeddings_output: location to where the embeddings should be written.
    stats_output: location to where the stats should be written.
    feature_key: tf.example feature that contains the samples that are to be
      processed.
    name: Identifier for the set of examples processed in this pipeline.
    batch_size: batch_size.

  Returns:
    The beam pipeline.
  """
  pipeline = beam.Pipeline()
  if files_input_list:
    examples = (
        pipeline
        | 'Read File List' >> ReadFromText(files_input_list)
        | 'Read Files' >> beam.ParDo(ReadWavFiles()))
  else:
    examples = (
        pipeline
        | 'Read Examples' >> ReadFromTFRecord(
            tfrecord_input,
            value_coder=beam.coders.ProtoCoder(tf.train.Example))
        | 'Add Keys' >> beam.ParDo(AddKey()))
  embeddings = (
      examples
      | 'Batched Inference' >> beam.ParDo(
          BatchedInference(
              batch_size=batch_size,
              model=embedding_model,
              feature_key=feature_key)).with_outputs('raw', main='examples'))
  if stats_output:
    _ = (
        embeddings.raw
        | 'Combine Embeddings' >> beam.CombineGlobally(
            ComputeMeanAndCovariance(key_name=name, embedding_dim=128))
        | 'DropKey' >> beam.ParDo(DropKey())
        | 'Write Stats' >> WriteToTFRecord(
            stats_output,
            shard_name_template='',
            coder=beam.coders.ProtoCoder(tf.train.Example)))
  if embeddings_output:
    _ = (
        embeddings.examples
        | 'DropKeyEmbeddings' >> beam.ParDo(DropKey())
        | 'Write Examples' >> WriteToTFRecord(
            embeddings_output,
            shard_name_template='',
            coder=beam.coders.ProtoCoder(tf.train.Example)))
  return pipeline
