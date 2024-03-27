# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""ENTRYPOINT for training and testing the model for video timeline modeling.

This code is PyTorch-based, which aims to run on XCloud with GPUs.
"""

import json
import os
from typing import List, Tuple

from absl import app
from absl import flags
from absl import logging
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from vtm.dataset import collate_topics
from vtm.dataset import TimelineDataset
from vtm.model.model import ClassifierModel
from vtm.model.model import TimelineModel

_FAIL_ON_CPU = flags.DEFINE_boolean('fail_on_cpu', False,
                                    'fail if not run on GPU')
_DATA_PATH = flags.DEFINE_string('data_path', None, 'The dataset path.')
_CHECKPOINT_DIR = flags.DEFINE_string('checkpoint_dir', None,
                                      'Directory for saving checkpoints.')
_TENSORBOARD_DIR = flags.DEFINE_string(
    'tensorboard_dir', None, 'Directory for saving tensorboard events.')
_TRAINED_MODEL_PATH = flags.DEFINE_string(
    'trained_model_path', None,
    'If given, only run inference using the trained model.')

# Problem hyperparas
_MAX_NUM_CLUSTER = flags.DEFINE_integer('max_num_cluster', 24,
                                        'max number of clusters')
_MAX_NUM_VIDEO = flags.DEFINE_integer('max_num_video', 120,
                                      'max number of videos')
_VIDEO_FEATURE = flags.DEFINE_string('video_feature',
                                     'vca_video_features_pulsar_embedding',
                                     'The used video input feature.')
_OFFLINE_DISTILLATION = flags.DEFINE_boolean(
    'offline_distillation', False,
    ('Apply knowledge distillation if True.'
     'The teacher model is the model with text embeddings as input.'))
_TRAINED_TEACHER_MODEL_PATH = flags.DEFINE_string(
    'trained_teacher_model_path', None,
    'The pretrained teacher model path, used for distillation.')
_ONLINE_DISTILLATION = flags.DEFINE_boolean(
    'online_distillation', False,
    ('Apply online knowledge distillation if True.'
     'The teacher model is the model with text embeddings as input.'))
_FEATURE_DISTILLATION = flags.DEFINE_boolean(
    'feature_distillation', False,
    ('Distill the intermediate features if True.'
     'The teacher model is the model with text embeddings as input.'))

# Model hyperparas
_RUN_BASELINE = flags.DEFINE_boolean(
    'run_baseline', False,
    'Run the baseline model if True; otherwise run our model.')
_REMOVE_VIDEO_AND_CLUSTER_ENCODERS = flags.DEFINE_boolean(
    'remove_video_and_cluster_encoders', False,
    'Remove video and cluster corresponding encoders if True.')
_SEMANTICS_AWARE_HEAD = flags.DEFINE_boolean(
    'semantics_aware_head', False, 'Add the semantics-aware head if True.')
_CONTRASTIVE_LOSS = flags.DEFINE_boolean(
    'contrastive_loss', False,
    ('Use contrastive loss for the semantics-aware head if True.'
     'Otherwise, use the consine similarity loss.'))
_TEMPERATURE = flags.DEFINE_float(
    'temperature', 0.07, 'Temperature value used for contrastive loss.')
_SEMANTICS_AWARE_HEAD_POS = flags.DEFINE_enum(
    'semantics_aware_head_pos', 'pos1', ['pos1', 'pos2'],
    'The position to place the semantics-aware head.')
_TEXT_EMBEDDING_AS_INPUT = flags.DEFINE_boolean(
    'text_embedding_as_input', False,
    'Include the text embeddings as input if True.')
_NUM_EMB = flags.DEFINE_integer(
    'num_emb', 256,
    'number of hidden dimensions for learnable cluster embeddings')
_NUM_INPUT_HIDDEN_VIDEO = flags.DEFINE_integer(
    'num_input_hidden_video', 256,
    'number of hidden dimensions for input video embeddings')
_NUM_HIDDEN = flags.DEFINE_integer(
    'num_hidden', 256, 'number of hidden dimensions in Transformer encoders')
_NUM_HEAD = flags.DEFINE_integer(
    'num_head', 2, 'number of attention heads in Transformer encoders')
_NUM_LAYERS = flags.DEFINE_integer('num_layers', 1,
                                   'number of layers in Transformer encoders')
_VIDEO_PE = flags.DEFINE_boolean('video_pe', False,
                                 'if apply positional encoding to videos')
_DROPOUT = flags.DEFINE_float('dropout', 0.1,
                              'dropout rate in Transformer encoders')
_SEMANTICS_LOSS_WEIGHT = flags.DEFINE_float(
    'semantics_loss_weight', 1, 'the weight for the semantics-aware head loss')
_DISTILLATION_LOSS_WEIGHT = flags.DEFINE_float(
    'distillation_loss_weight', 0.1, 'the weight for the distillation loss')
_TEACHER_LOSS_WEIGHT = flags.DEFINE_float(
    'teacher_loss_weight', 1,
    'the weight for the teacher model loss during online distillation')

# Training hyperparas
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 16, 'batch size')
_EPOCHS = flags.DEFINE_integer('epochs', 10, 'epochs')
_LOG_STEPSIZE = flags.DEFINE_integer('log_stepsize', 100, 'log step size')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
_WEIGHT_DECAY = flags.DEFINE_float('weight_decay', 0.00001, 'weight decay')


def check_gpu():
  """Print GPU info and return 'cuda' if found, 'cpu' otherwise."""
  try:
    logging.info('FLAGS.fail_on_cpu: %s', _FAIL_ON_CPU.value)
    logging.info('torch.__version__: %s', torch.__version__)
    logging.info('torch.cuda.device_count(): %s', torch.cuda.device_count())
    logging.info('torch.cuda.current_device(): %s', torch.cuda.current_device())
    logging.info('torch.cuda.get_device_name(0): %s',
                 torch.cuda.get_device_name(0))
    logging.info('torch.cuda.is_available(0): %s', torch.cuda.is_available())
    if torch.cuda.is_available():
      return 'cuda'
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(e)
  if _FAIL_ON_CPU.value:
    logging.error('Not able to run on CPU')
    exit(1)
  logging.error('Falling back to CPU.')
  return 'cpu'


# We cannot use built-in types (tuple/list/dict) for parametric annotations (
# supported by python >=3.9), since the xcloud pre-built PyTorch image only
# supports up to python 3.8.
def get_dataset():
  """Initialize datasets."""
  train_dataset = TimelineDataset(
      partition='train',
      feature_key=_VIDEO_FEATURE.value,
      feature_dim=_NUM_INPUT_HIDDEN_VIDEO.value,
      data_path=_DATA_PATH.value)
  valid_dataset = TimelineDataset(
      partition='valid',
      feature_key=_VIDEO_FEATURE.value,
      feature_dim=_NUM_INPUT_HIDDEN_VIDEO.value,
      data_path=_DATA_PATH.value)
  test_dataset = TimelineDataset(
      partition='test',
      feature_key=_VIDEO_FEATURE.value,
      feature_dim=_NUM_INPUT_HIDDEN_VIDEO.value,
      data_path=_DATA_PATH.value)
  return train_dataset, valid_dataset, test_dataset


def train_epoch(timeline_model,
                device,
                train_loader,
                optimizer,
                teacher_model = None):
  """Training loop for one epoch."""
  timeline_model.train()
  train_loss = 0
  total_video = 0
  if _SEMANTICS_AWARE_HEAD.value:
    train_semantics_loss = 0
    total_clusters = 0
  if _OFFLINE_DISTILLATION.value:
    teacher_model.eval()
    train_distillation_loss = 0
    total_clusters = 0
  if _ONLINE_DISTILLATION.value:
    teacher_model.train()
    train_distillation_loss = 0
    train_teacher_model_loss = 0
    total_clusters = 0
  for step, data_batch in enumerate(train_loader):
    for key in data_batch:
      data_batch[key] = data_batch[key].to(device)
    optimizer.zero_grad()
    if _SEMANTICS_AWARE_HEAD.value:
      log_score, cluster_semantics_h, _ = timeline_model(data_batch)
    else:
      log_score, cluster_intermediate_h, video_intermediate_h = timeline_model(
          data_batch)
    # Only compute loss for non-padding video tokens
    log_score = log_score.view(-1, _MAX_NUM_CLUSTER.value)
    video_cluster_label = data_batch['video_cluster_label'].view(-1)
    video_non_padding_mask = ~data_batch['video_padding_mask'].view(-1)
    loss = F.nll_loss(
        log_score[video_non_padding_mask],
        video_cluster_label[video_non_padding_mask],
        reduction='sum')

    # With semantics-aware head: compute loss for non-padding cluster tokens
    if _SEMANTICS_AWARE_HEAD.value:
      if _CONTRASTIVE_LOSS.value:
        # (max_num_cluster, num_emb, batch_size)
        cluster_semantics_h = cluster_semantics_h.permute((1, 2, 0))
        # (max_num_cluster, max_num_cluster, batch_size)
        cosine_similarity_pairwise = F.cosine_similarity(
            cluster_semantics_h, cluster_semantics_h.unsqueeze(1), dim=-2)
        # (batch_size, max_num_cluster, max_num_cluster)
        cosine_similarity_pairwise = cosine_similarity_pairwise.permute(
            (2, 0, 1)) / _TEMPERATURE.value
        # (batch_size, max_num_cluster)
        self_cosine_similarity = torch.diagonal(
            cosine_similarity_pairwise, dim1=-2, dim2=-1)
        semantics_loss = (
            -self_cosine_similarity[data_batch['cluster_non_padding_mask']] +
            torch.logsumexp(
                cosine_similarity_pairwise[
                    data_batch['cluster_non_padding_mask']],
                dim=-1)).sum()
      else:
        semantics_loss = (1 - F.cosine_similarity(
            cluster_semantics_h[data_batch['cluster_non_padding_mask']],
            data_batch['cluster_text_features'][
                data_batch['cluster_non_padding_mask']])).sum()
      loss_sum = loss + _SEMANTICS_LOSS_WEIGHT.value * semantics_loss
      loss_sum.backward()
      optimizer.step()
      if step % _LOG_STEPSIZE.value == 0:
        logging.info('[%s/%s] Loss: %s',
                     step * len(data_batch['video_features']),
                     len(train_loader.dataset),
                     loss.item() / video_non_padding_mask.sum().item())
        logging.info(
            '[%s/%s] Semantics Loss: %s',
            step * len(data_batch['video_features']), len(train_loader.dataset),
            semantics_loss.item() /
            data_batch['cluster_non_padding_mask'].sum().item())
      train_semantics_loss += semantics_loss.item()
      train_loss += loss.item()
      total_video += video_non_padding_mask.sum().item()
      total_clusters += data_batch['cluster_non_padding_mask'].sum().item()

    # Offline knowledge distillation
    elif _OFFLINE_DISTILLATION.value:
      with torch.no_grad():
        teacher_log_score, teacher_cluster_h, teacher_video_h = teacher_model(
            data_batch)
      teacher_log_score = teacher_log_score.view(-1, _MAX_NUM_CLUSTER.value)
      if _FEATURE_DISTILLATION.value:
        distillation_cluster_loss = torch.norm(
            teacher_cluster_h - cluster_intermediate_h, dim=-1).sum()
        distillation_video_loss = torch.norm(
            teacher_video_h[~data_batch['video_padding_mask']] -
            video_intermediate_h[~data_batch['video_padding_mask']],
            dim=-1).sum()
        distillation_loss = distillation_cluster_loss + distillation_video_loss
      else:
        distillation_loss = F.kl_div(
            log_score[video_non_padding_mask],
            teacher_log_score[video_non_padding_mask],
            reduction='sum',
            log_target=True)
      loss_sum = loss + _DISTILLATION_LOSS_WEIGHT.value * distillation_loss
      loss_sum.backward()
      optimizer.step()
      if step % _LOG_STEPSIZE.value == 0:
        logging.info('[%s/%s] Loss: %s',
                     step * len(data_batch['video_features']),
                     len(train_loader.dataset),
                     loss.item() / video_non_padding_mask.sum().item())
        logging.info(
            '[%s/%s] Distillation Loss: %s',
            step * len(data_batch['video_features']), len(train_loader.dataset),
            distillation_loss.item() /
            (video_non_padding_mask.sum().item() +
             torch.numel(data_batch['cluster_non_padding_mask'])))
      train_distillation_loss += distillation_loss.item()
      train_loss += loss.item()
      total_video += video_non_padding_mask.sum().item()
      total_clusters += data_batch['cluster_non_padding_mask'].sum().item()

    # Online knowledge distillation
    elif _ONLINE_DISTILLATION.value:
      teacher_log_score, teacher_cluster_h, teacher_video_h = teacher_model(
          data_batch)
      teacher_log_score = teacher_log_score.view(-1, _MAX_NUM_CLUSTER.value)
      if _FEATURE_DISTILLATION.value:
        distillation_cluster_loss = torch.norm(
            teacher_cluster_h - cluster_intermediate_h, dim=-1).sum()
        distillation_video_loss = torch.norm(
            teacher_video_h[~data_batch['video_padding_mask']] -
            video_intermediate_h[~data_batch['video_padding_mask']],
            dim=-1).sum()
        distillation_loss = distillation_cluster_loss + distillation_video_loss
      else:
        distillation_loss = F.kl_div(
            log_score[video_non_padding_mask],
            teacher_log_score[video_non_padding_mask],
            reduction='sum',
            log_target=True)
      teacher_model_loss = F.nll_loss(
          teacher_log_score[video_non_padding_mask],
          video_cluster_label[video_non_padding_mask],
          reduction='sum')
      loss_sum = (loss + _DISTILLATION_LOSS_WEIGHT.value * distillation_loss
                  + _TEACHER_LOSS_WEIGHT.value * teacher_model_loss)
      loss_sum.backward()
      optimizer.step()
      if step % _LOG_STEPSIZE.value == 0:
        logging.info('[%s/%s] Loss: %s',
                     step * len(data_batch['video_features']),
                     len(train_loader.dataset),
                     loss.item() / video_non_padding_mask.sum().item())
        logging.info(
            '[%s/%s] Teacher Model Loss: %s',
            step * len(data_batch['video_features']), len(train_loader.dataset),
            teacher_model_loss.item() / video_non_padding_mask.sum().item())
        logging.info(
            '[%s/%s] Distillation Loss: %s',
            step * len(data_batch['video_features']), len(train_loader.dataset),
            distillation_loss.item() /
            (video_non_padding_mask.sum().item() +
             torch.numel(data_batch['cluster_non_padding_mask'])))
      train_distillation_loss += distillation_loss.item()
      train_loss += loss.item()
      train_teacher_model_loss += teacher_model_loss.item()
      total_video += video_non_padding_mask.sum().item()
      total_clusters += data_batch['cluster_non_padding_mask'].sum().item()

    else:
      loss.backward()
      optimizer.step()
      if step % _LOG_STEPSIZE.value == 0:
        logging.info('[%s/%s] Loss: %s',
                     step * len(data_batch['video_features']),
                     len(train_loader.dataset),
                     loss.item() / video_non_padding_mask.sum().item())
      train_loss += loss.item()
      total_video += video_non_padding_mask.sum().item()

  if _SEMANTICS_AWARE_HEAD.value:
    return train_loss / total_video, train_semantics_loss / total_clusters, None
  elif _OFFLINE_DISTILLATION.value:
    return train_loss / total_video, train_distillation_loss / (
        total_video + total_clusters), None
  elif _ONLINE_DISTILLATION.value:
    return train_loss / total_video, train_distillation_loss / (
        total_video + total_clusters), train_teacher_model_loss / total_video,
  else:
    return train_loss / total_video, None, None


def evaluate(timeline_model, device,
             loader):
  """Evaluation pipeline for measuring video to cluster accuracy (float)."""
  timeline_model.eval()
  video_to_cluster_correct = 0
  total_video = 0
  with torch.no_grad():
    for data_batch in loader:
      for key in [
          'video_features', 'video_padding_mask', 'video_cluster_label'
      ]:
        data_batch[key] = data_batch[key].to(device)
      log_score, _, _ = timeline_model(data_batch)
      log_score = log_score.view(-1, _MAX_NUM_CLUSTER.value)
      prediction = log_score.argmax(dim=1, keepdim=False)
      video_cluster_label = data_batch['video_cluster_label'].view(-1)
      video_non_padding_mask = ~data_batch['video_padding_mask'].view(-1)
      video_to_cluster_correct += prediction[video_non_padding_mask].eq(
          video_cluster_label[video_non_padding_mask]).sum().item()
      total_video += video_non_padding_mask.sum().item()
  video_to_cluster_accuracy = video_to_cluster_correct / total_video
  return video_to_cluster_accuracy


def inference(timeline_model, device,
              dataset):
  """Inference with one-by-one processing."""
  timeline_model.eval()
  video_to_cluster_correct = 0
  total_video = 0
  final_predictions = []
  loader = DataLoader(
      dataset, batch_size=1, shuffle=False, collate_fn=collate_topics)
  with torch.no_grad():
    for i, data_batch in enumerate(loader):
      data_prediction = {}
      for key in [
          'video_features', 'video_padding_mask', 'video_cluster_label'
      ]:
        data_batch[key] = data_batch[key].to(device)
      log_score, _, _ = timeline_model(data_batch)
      log_score = log_score.view(-1, _MAX_NUM_CLUSTER.value)
      prediction = log_score.argmax(dim=1, keepdim=False)
      video_cluster_label = data_batch['video_cluster_label'].view(-1)
      video_non_padding_mask = ~data_batch['video_padding_mask'].view(-1)
      video_to_cluster_correct += prediction[video_non_padding_mask].eq(
          video_cluster_label[video_non_padding_mask]).sum().item()
      total_video += video_non_padding_mask.sum().item()
      data_prediction['timeline_url'] = dataset[i]['timeline_url']
      data_prediction['pred'] = prediction.tolist()
      data_prediction['label'] = video_cluster_label.tolist()
      final_predictions.append(data_prediction)

  video_to_cluster_accuracy = video_to_cluster_correct / total_video
  return final_predictions, video_to_cluster_accuracy


def save_model(model, optimizer, output_dir):
  """Save model to GCS."""
  os.makedirs(output_dir, exist_ok=True)
  # Will overwrite existing previously saved model.
  torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
  torch.save(
      {
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
      }, os.path.join(output_dir, 'checkpoint.tar'))


def main(_):
  logging.info('Job started')
  device = torch.device(check_gpu())
  torch.cuda.empty_cache()

  logging.info('FLAGS.epochs: %s', _EPOCHS.value)
  logging.info('FLAGS.batch_size: %s', _BATCH_SIZE.value)
  logging.info('FLAGS.learning_rate: %s', _LEARNING_RATE.value)
  logging.info('FLAGS.weight_decay: %s', _WEIGHT_DECAY.value)

  if _RUN_BASELINE.value:
    logging.info('Running baseline model.')
    model = ClassifierModel(_MAX_NUM_CLUSTER.value, _MAX_NUM_VIDEO.value,
                            _NUM_EMB.value, _NUM_INPUT_HIDDEN_VIDEO.value,
                            _NUM_HIDDEN.value, _NUM_HEAD.value,
                            _NUM_LAYERS.value, _VIDEO_PE.value, _DROPOUT.value)
  else:
    if _SEMANTICS_AWARE_HEAD.value:
      logging.info('Running our complete model.')
    else:
      if _REMOVE_VIDEO_AND_CLUSTER_ENCODERS.value:
        logging.info('Running our model without the semantics-aware head.')
        logging.info('and the video and cluster encoders.')
      else:
        logging.info('Running our model without the semantics-aware head.')
    model = TimelineModel(_MAX_NUM_CLUSTER.value, _MAX_NUM_VIDEO.value,
                          _NUM_EMB.value, _NUM_INPUT_HIDDEN_VIDEO.value,
                          _NUM_HIDDEN.value, _NUM_HEAD.value, _NUM_LAYERS.value,
                          _VIDEO_PE.value, _DROPOUT.value,
                          _SEMANTICS_AWARE_HEAD.value,
                          _SEMANTICS_AWARE_HEAD_POS.value,
                          _REMOVE_VIDEO_AND_CLUSTER_ENCODERS.value,
                          _TEXT_EMBEDDING_AS_INPUT.value)
    if _OFFLINE_DISTILLATION.value or _ONLINE_DISTILLATION.value:
      assert not _SEMANTICS_AWARE_HEAD.value
      assert not _REMOVE_VIDEO_AND_CLUSTER_ENCODERS.value
      teacher_model = TimelineModel(
          _MAX_NUM_CLUSTER.value, _MAX_NUM_VIDEO.value, _NUM_EMB.value,
          _NUM_INPUT_HIDDEN_VIDEO.value, _NUM_HIDDEN.value, _NUM_HEAD.value,
          _NUM_LAYERS.value, _VIDEO_PE.value, _DROPOUT.value,
          _SEMANTICS_AWARE_HEAD.value, _SEMANTICS_AWARE_HEAD_POS.value,
          _REMOVE_VIDEO_AND_CLUSTER_ENCODERS.value, True)
      teacher_model = nn.DataParallel(teacher_model).to(device)
      if _OFFLINE_DISTILLATION.value:
        logging.info('Performing offline knowledge distillation.')
        teacher_model.load_state_dict(
            torch.load(
                os.path.join(_TRAINED_TEACHER_MODEL_PATH.value, 'model.pt')))
        logging.info('Loaded pretrained teacher model.')
      elif _ONLINE_DISTILLATION.value:
        logging.info('Performing online knowledge distillation.')
  train_dataset, valid_dataset, test_dataset = get_dataset()
  trained_model_path = _TRAINED_MODEL_PATH.value
  if trained_model_path is not None:
    logging.info('Run inference only.')
    timeline_model = nn.DataParallel(model).to(device)
    timeline_model.load_state_dict(
        torch.load(os.path.join(trained_model_path, 'model.pt')))
    valid_predictions, final_valid_v2c_acc = inference(timeline_model, device,
                                                       valid_dataset)
    test_predictions, final_test_v2c_acc = inference(timeline_model, device,
                                                     test_dataset)
    logging.info('Final Valid Acc %s', final_valid_v2c_acc)
    logging.info('Final Test Acc %s', final_test_v2c_acc)
    with open(os.path.join(trained_model_path, 'valid_prediction.json'),
              'w') as f:
      json.dump(valid_predictions, f)
    with open(os.path.join(trained_model_path, 'test_prediction.json'),
              'w') as f:
      json.dump(test_predictions, f)
  else:
    timeline_model = nn.DataParallel(model).to(device)

    if _OFFLINE_DISTILLATION.value or _ONLINE_DISTILLATION.value:
      optimizer = optim.Adam(
          list(timeline_model.parameters()) + list(teacher_model.parameters()),
          lr=_LEARNING_RATE.value,
          weight_decay=_WEIGHT_DECAY.value)
    else:
      optimizer = optim.Adam(
          timeline_model.parameters(),
          lr=_LEARNING_RATE.value,
          weight_decay=_WEIGHT_DECAY.value)

    if _TENSORBOARD_DIR.value:
      writer = tensorboard.SummaryWriter(_TENSORBOARD_DIR.value)

    train_loader = DataLoader(
        train_dataset,
        batch_size=_BATCH_SIZE.value,
        shuffle=True,
        collate_fn=collate_topics)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=_BATCH_SIZE.value,
        shuffle=False,
        collate_fn=collate_topics)
    test_loader = DataLoader(
        test_dataset,
        batch_size=_BATCH_SIZE.value,
        shuffle=False,
        collate_fn=collate_topics)

    best_valid_v2c_acc = 0
    for epoch in range(1, _EPOCHS.value + 1):
      logging.info('Epoch %s of %s', epoch, _EPOCHS.value)
      if _SEMANTICS_AWARE_HEAD.value:
        train_loss, semantics_loss, _ = train_epoch(timeline_model, device,
                                                    train_loader, optimizer)
        logging.info('Loss %s', train_loss)
        logging.info('Semantics loss %s', semantics_loss)
      elif _OFFLINE_DISTILLATION.value:
        train_loss, distillation_loss, _ = train_epoch(timeline_model, device,
                                                       train_loader, optimizer,
                                                       teacher_model)
        logging.info('Loss %s', train_loss)
        logging.info('Distillation loss %s', distillation_loss)
      elif _ONLINE_DISTILLATION.value:
        train_loss, distillation_loss, train_teacher_model_loss = train_epoch(
            timeline_model, device, train_loader, optimizer, teacher_model)
        logging.info('Loss %s', train_loss)
        logging.info('Teacher Model Loss %s', train_teacher_model_loss)
        logging.info('Distillation loss %s', distillation_loss)
      else:
        train_loss, _, _ = train_epoch(timeline_model, device, train_loader,
                                       optimizer)
        logging.info('Loss %s', train_loss)
      train_v2c_acc = evaluate(timeline_model, device, train_loader)
      valid_v2c_acc = evaluate(timeline_model, device, valid_loader)
      test_v2c_acc = evaluate(timeline_model, device, test_loader)
      if valid_v2c_acc > best_valid_v2c_acc:
        best_valid_v2c_acc = valid_v2c_acc
        final_test_v2c_acc = test_v2c_acc
        if _CHECKPOINT_DIR.value:
          save_model(timeline_model, optimizer, _CHECKPOINT_DIR.value)
      logging.info('Training Acc %s', train_v2c_acc)
      logging.info('Valid Acc %s', valid_v2c_acc)
      logging.info('Test Acc %s', test_v2c_acc)
      logging.info('Best Valid Acc So Far %s', best_valid_v2c_acc)
      logging.info('Final Test Acc So Far %s', final_test_v2c_acc)

      if _TENSORBOARD_DIR.value:
        if _SEMANTICS_AWARE_HEAD.value:
          writer.add_scalar('Semantics Loss/train', semantics_loss, epoch)
        if _OFFLINE_DISTILLATION.value:
          writer.add_scalar('Distillation Loss/train', distillation_loss, epoch)
        if _ONLINE_DISTILLATION.value:
          writer.add_scalar('Distillation Loss/train', distillation_loss, epoch)
          writer.add_scalar('Teacher Model Loss/train',
                            train_teacher_model_loss, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_v2c_acc, epoch)
        writer.add_scalar('Accuracy/valid', valid_v2c_acc, epoch)
        writer.add_scalar('Accuracy/test', test_v2c_acc, epoch)
        logging.info('Flushing TensorBoard writer')
        writer.flush()

    if _TENSORBOARD_DIR.value:
      writer.close()

  logging.info('Job finished')


if __name__ == '__main__':
  app.run(main)
