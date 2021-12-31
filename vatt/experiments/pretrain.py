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

# Lint as: python3
"""Main experiments script for pre-training MMV model OR vatt."""

from typing import Optional

from absl import logging
import numpy as np
import tensorflow as tf

from vatt.data import dataloaders
from vatt.data import processing
from vatt.experiments import base
from vatt.modeling import factory as model_factory
from vatt.utils.eval import evaluators
from vatt.utils.train import optimizers
from vatt.utils.train import restore
from vatt.utils.train import schedules


FeatureNames = dataloaders.FeatureNames
REF_FPS = dataloaders.REF_FPS  # reference fps used during training
REF_SR = dataloaders.REF_SR  # reference sampling rate used during training


class BaseExecutor(base.Executor):
  """Constructs the necessary modules to perform train or evaluation."""

  def __init__(self, params):
    strategy = base.create_strategy(params.strategy_config)
    data = self.construct_data(params)
    with strategy.scope():
      model = self.construct_model(params)

    super(BaseExecutor, self).__init__(model=model,
                                       data=data,
                                       params=params,
                                       strategy=strategy,
                                       metrics=None)

  def restore_text_embeddings(self, model, params):
    """Partially restore weights (text embeddings currently)."""

    model_name = params.model_config.model_name
    backbone_cfg_mode = params.model_config.backbone_config.name
    vat_layer = model.get_layer(model_name).get_layer(backbone_cfg_mode)
    if backbone_cfg_mode.startswith('backbone_stack'):
      embedding_layer = vat_layer.txt_backbone.embedding
    elif backbone_cfg_mode.startswith('unified_backbone'):
      embedding_layer = (
          vat_layer.unified_backbone.unified_transformer
          .raw_to_embeddings['text']
          )

    tokenizer = params.train.input.text_tokenizer
    if tokenizer == 'WordTokenizer':
      embedding_name = 'word2vec'
      embedding_vocab_size = 2**16
    elif tokenizer == 'BertTokenizer':
      d_model = embedding_layer.output_dim
      embedding_vocab_size = 30522
      dim2size = {512: 'small', 768: 'base', 1024: 'large'}
      embedding_name = 'bert_uncased_{}'.format(dim2size[d_model])
    else:
      raise ValueError('Text tokenizer {!r} not supported!'.format(tokenizer))

    # make sure the correct vocab_size has been used
    assert embedding_layer.input_dim == embedding_vocab_size, (
        'Text embedding layer is not configured properly. '
        'Expected vocab_size={}, but configured with vocab_size={}.'
        .format(embedding_vocab_size, embedding_layer.input_dim)
        )

    # finally restore embedding weights
    restore.assign_word_embeddings(embedding_layer, embedding_name)
    logging.info('Language embedding weights %s restored successfully.',
                 embedding_name)

  def construct_model(self, params):
    """Build models for train/eval."""

    if params.mode == 'train':
      input_params = params.train.input
      space_to_depth = input_params.space_to_depth
    else:
      input_params = params.eval.input
      space_to_depth = input_params.space_to_depth

    video_shape = processing.get_video_shape(input_params, space_to_depth)
    audio_shape = processing.get_audio_shape(input_params, REF_FPS, REF_SR)
    text_shape = (input_params.max_num_words,)

    inputs = {
        'video': tf.keras.Input(shape=video_shape),
        'audio': tf.keras.Input(shape=audio_shape),
        'text': tf.keras.Input(shape=text_shape),
    }

    model = model_factory.build_model(params.model_config)
    outputs = model(inputs, None)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.loss_fn = model.loss_fn

    # Restoring word embeddings
    self.restore_text_embeddings(keras_model, params)

    logging.info('Number of parameters in model: %f M.',
                 keras_model.count_params() / 10.**6)

    learning_rate = schedules.get_learning_rate(
        params.train.optimizer.learning_rate
        )
    keras_model.optimizer = optimizers.get_optimizer(
        learning_rate, params.train.optimizer
        )
    return keras_model

  def construct_data(self, params):
    if params.mode == 'train':
      data = [dataloaders.PreTrainLoader(dataset_id=params.train.input.name,
                                         params=params)]
    elif params.mode == 'eval':
      data = []
      for dataset_id in params.eval.input.name:
        if dataset_id in dataloaders.CLS_DS:
          for subset in ['train', 'test']:
            for split in dataloaders.CLS_DS[dataset_id]['splits']:
              data.append(dataloaders.EvalLoader(
                  dataset_id=dataset_id,
                  subset=subset,
                  params=params,
                  split=split,
                  ))
        else:
          data.append(dataloaders.EvalLoader(
              dataset_id=dataset_id,
              subset='test',
              params=params,
              ))
    else:
      raise ValueError('Invalid mode!')

    return data


class TrainExecutor(BaseExecutor):
  """Constructs the necessary modules to perform training."""

  def prepare_inputs(self, inputs):
    """Prepares inputs on device to be fed to model in train mode."""

    params = self.params.train.input
    images = inputs[FeatureNames.VISION]
    space_to_depth = params.space_to_depth
    if params.linearize_vision:
      vid_shape = processing.get_video_shape(params,
                                             is_space_to_depth=space_to_depth)
      images = tf.reshape(images, [-1] + vid_shape)

    if params.raw_audio:
      audio = inputs[FeatureNames.AUDIO]
    else:
      audio = inputs[FeatureNames.AUDIO_MEL]
    words = inputs[FeatureNames.TEXT_INDEX]
    words = tf.reshape(words, [-1, words.shape.as_list()[-1]])

    audio_mask = inputs[FeatureNames.AUDIO_MASK]
    text_mask = inputs[FeatureNames.TEXT_MASK]

    labels = {FeatureNames.AUDIO_MASK: audio_mask,
              FeatureNames.TEXT_MASK: text_mask}

    inputs = {'video': images,
              'audio': audio,
              'text': words}
    return inputs, labels


class EvalExecutor(BaseExecutor):
  """Constructs the necessary modules to perform evaluation."""

  def prepare_inputs(self, inputs):
    """Prepares inputs on device to be fed to model in eval mode."""

    params = self.params.eval.input
    video_shape = processing.get_video_shape(params)
    audio_shape = processing.get_audio_shape(params, REF_FPS, REF_SR)

    if FeatureNames.VISION in inputs:
      images = inputs[FeatureNames.VISION]
    else:
      images = tf.zeros([1] + video_shape, dtype=tf.float32)

    if FeatureNames.AUDIO_MEL in inputs or FeatureNames.AUDIO in inputs:
      if params.raw_audio:
        audio = inputs[FeatureNames.AUDIO]
      else:
        audio = inputs[FeatureNames.AUDIO_MEL]
    else:
      audio = tf.zeros([1] + audio_shape, dtype=tf.float32)

    if FeatureNames.TEXT_INDEX in inputs:
      words = inputs[FeatureNames.TEXT_INDEX]
    else:
      words = tf.zeros([1, params.max_num_words], dtype=tf.int32)

    audio = tf.reshape(audio, [-1] + audio_shape)
    words = tf.reshape(words, [-1, words.shape.as_list()[-1]])

    labels_onehot = inputs.get(FeatureNames.LABEL_INDEX, None)

    labels = {'one_hot': labels_onehot}

    inputs = {'video': images,
              'audio': audio,
              'text': words}

    return inputs, labels

  def _create_outputs_filter(self, task, modality = None):
    """."""

    if task == 'classification':
      assert modality is not None, 'Modality should be provided'
      def outputs_filter(outputs):
        labels = outputs['labels']
        outputs = {'features': outputs[modality]['features_pooled'],
                   'labels': tf.argmax(labels['one_hot'], axis=1)[:, None]}
        return outputs

    elif task == 'retrieval':
      def outputs_filter(outputs):
        vid_embd = outputs['head_stack']['bridge']['video']
        aud_embd = outputs['head_stack']['bridge']['audio']
        txt_embd = outputs['head_stack']['bridge']['text']
        outputs = {'test_vid2txt_embd': vid_embd['totxt'],
                   'test_vid2aud_embd': vid_embd['toaud'],
                   'test_aud2vid_embd': aud_embd['tovid'],
                   'test_txt2vid_embd': txt_embd['tovid']}

        return outputs

    else:
      raise NotImplementedError

    return outputs_filter

  def _run_linear_classification(self,
                                 dataset_id,
                                 dataset_split,
                                 train_iterator,
                                 test_iterator):
    """Runs offline linear classification."""

    params = self.params
    strategy = self.strategy

    test_input_params = params.eval.input
    n_test_windows = test_input_params.num_windows_test
    num_train_batch_clips = test_input_params.batch_size
    num_test_batch_clips = test_input_params.batch_size * n_test_windows
    if test_input_params.multi_crop:
      num_test_batch_clips = num_test_batch_clips * 3

    if dataset_id in dataloaders.VID_CLS_DS:
      modality = 'video'
    elif dataset_id in dataloaders.AUD_CLS_DS:
      modality = 'audio'

    outputs_filter = self._create_outputs_filter(task='classification',
                                                 modality=modality)

    train_outputs, cnt = self.infer(iterator=train_iterator,
                                    outputs_filter=outputs_filter)
    logging.info('Finished model inference on %s training clips.',
                 cnt * num_train_batch_clips)

    test_outputs, cnt = self.infer(iterator=test_iterator,
                                   outputs_filter=outputs_filter)
    logging.info('Finished model inference on %s testing clips.',
                 cnt * num_test_batch_clips)

    # aggregate all steps
    for k in train_outputs:
      train_outputs[k] = np.concatenate(train_outputs[k], axis=0)
    for k in test_outputs:
      test_outputs[k] = np.concatenate(test_outputs[k], axis=0)

    train_metrics, test_metrics = evaluators.linear_classifier(
        train_features=train_outputs['features'],
        test_features=test_outputs['features'],
        train_labels=train_outputs['labels'],
        test_labels=test_outputs['labels'],
        dataset_id=dataset_id,
        num_windows_test=n_test_windows,
        strategy=strategy,
        )
    logging.info('Classification results:\n Train: %r\n Test: %r',
                 train_metrics, test_metrics)

    metrics = {'_'.join([dataset_id,
                         dataset_split,
                         'train',
                         'top_1_accuracy']): train_metrics['top1'],
               '_'.join([dataset_id,
                         dataset_split,
                         'train',
                         'top_5_accuracy']): train_metrics['top5']}
    metrics.update({'_'.join([dataset_id,
                              dataset_split,
                              'test',
                              'top_1_accuracy']): test_metrics['top1'],
                    '_'.join([dataset_id,
                              dataset_split,
                              'test',
                              'top_5_accuracy']): test_metrics['top5']})

    return metrics

  def _run_zero_shot_retrieval(self, dataset_id, test_iterator):
    """Runs zero-shot cross-modal retrieval."""

    input_params = self.params.eval.input
    num_clips = dataloaders.TEXT_DS[dataset_id]['num_clips']
    num_steps = (num_clips // input_params.batch_size)
    n_windows = input_params.num_windows_test
    num_batch_clips = input_params.batch_size * n_windows
    assert num_steps > 0 and num_batch_clips > 0

    logging.info('Number of zero-shot testing steps %d', num_steps)

    outputs_filter = self._create_outputs_filter(task='retrieval')

    inference_outputs, cnt = self.infer(iterator=test_iterator,
                                        outputs_filter=outputs_filter,
                                        num_steps=num_steps)

    logging.info('Finished model inference on %s clips.', cnt * num_batch_clips)
    # aggregate all steps
    for k in inference_outputs:
      inference_outputs[k] = np.concatenate(inference_outputs[k], axis=0)

    # get similarities
    has_text = dataset_id in dataloaders.TEXT_DS
    has_audio = dataset_id in dataloaders.AUDIO_DS

    zs_metric_result = evaluators.modality_similarity(
        inference_outputs,
        has_text,
        has_audio,
        n_windows,
        )

    # Add the loader name in the metrics names.
    test_metric_result = {}
    for k in zs_metric_result:
      test_metric_result[dataset_id + '_' + k] = zs_metric_result[k]

    # update test_summary_writer
    logging.info('Retrieval results: %r', test_metric_result)

    return test_metric_result

  def _avg_cls_metrics(self, cls_metrics, data_loaders):
    """Takes average of classification metrics over different splits."""
    all_cls_metrics = {}
    for data_loader_name in data_loaders:
      dataset_id = data_loader_name.split('@')[0]
      is_cls = dataset_id in dataloaders.CLS_DS
      if is_cls:
        dataset_split = data_loader_name.split('@')[1]
        split_k_1 = '_'.join(
            [dataset_id, dataset_split, 'test', 'top_1_accuracy'])
        split_k_5 = '_'.join(
            [dataset_id, dataset_split, 'test', 'top_5_accuracy'])
        avg_k_1 = '_'.join([dataset_id, 'top_1_accuracy'])
        avg_k_5 = '_'.join([dataset_id, 'top_5_accuracy'])
        if avg_k_1 not in all_cls_metrics:
          all_cls_metrics[avg_k_1] = [cls_metrics[split_k_1]]
        else:
          all_cls_metrics[avg_k_1].append(cls_metrics[split_k_1])

        if avg_k_5 not in all_cls_metrics:
          all_cls_metrics[avg_k_5] = [cls_metrics[split_k_5]]
        else:
          all_cls_metrics[avg_k_5].append(cls_metrics[split_k_5])

    avg_cls_metrics = {}
    for k in all_cls_metrics:
      avg_cls_metrics[k] = np.mean(all_cls_metrics[k])

    return avg_cls_metrics

  def evaluation_loop(self):
    """Iterates over data and returns the aggregated metrics."""

    data = self.data
    strategy = self.strategy

    # organize dataloaders
    all_cls_metrics = {}
    all_ret_metrics = {}
    all_metrics = {}
    data_loader_dict = {}
    for data_loader in self.get_dataloaders(data, strategy):
      name = data_loader['name']
      mode = data_loader['mode']
      iterator = data_loader['iterator']
      if name not in data_loader_dict:
        data_loader_dict[name] = {mode: iterator}
      else:
        assert mode not in data_loader_dict[name], (
            'repetitive name-mode pairs')
        data_loader_dict[name].update({mode: iterator})

    for data_loader_name in data_loader_dict:
      dataset_id = data_loader_name.split('@')[0]
      is_cls = dataset_id in dataloaders.CLS_DS
      is_retrieval = not is_cls

      if is_cls:
        train_iterator = data_loader_dict[data_loader_name]['train']
        test_iterator = data_loader_dict[data_loader_name]['test']
        dataset_split = data_loader_name.split('@')[1]

        cls_metric_results = self._run_linear_classification(
            dataset_id,
            dataset_split,
            train_iterator,
            test_iterator,
            )
        all_cls_metrics.update(cls_metric_results)

      elif is_retrieval:
        test_iterator = data_loader_dict[data_loader_name]['test']
        logging.info('Testing zero-shot retrieval for %s started', dataset_id)
        ret_metric_results = self._run_zero_shot_retrieval(
            dataset_id,
            test_iterator,
            )
        all_ret_metrics.update(ret_metric_results)

    # average over all classification splits (test metrics)
    avg_cls_metrics = self._avg_cls_metrics(all_cls_metrics, data_loader_dict)

    all_metrics.update(all_cls_metrics)
    all_metrics.update(avg_cls_metrics)
    all_metrics.update(all_ret_metrics)

    return all_metrics


def get_executor(params):
  mode = params.mode
  if mode == 'train':
    return TrainExecutor(params=params)
  elif mode == 'eval':
    return EvalExecutor(params=params)
  else:
    raise ValueError('Invalid mode!')

