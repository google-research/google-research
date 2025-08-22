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

"""Train, test, and warmstart functions.

This file contains the training, testing, and warmstart functions which are used
in the main train and test scripts.
"""

import dataclasses
import datetime as dtfull
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric import data as torch_geo_data
from torch_geometric import loader as torch_geo_data_loader
import tqdm

from fm4tlp.models import model_template
from fm4tlp.modules import neighbor_loader
from fm4tlp.utils import evaluate
from fm4tlp.utils import negative_sampler


@dataclasses.dataclass(frozen=True)
class PerformanceMetricLists:
  loss: list[float] = dataclasses.field(default_factory=list)
  model_loss: list[float] = dataclasses.field(default_factory=list)
  perf: list[float] = dataclasses.field(default_factory=list)
  auc: list[float] = dataclasses.field(default_factory=list)


def normalize_structural_features(
    structural_feat, structural_feature_mean, structural_feature_std
):
  """Normalizes the structural features.

  Arguments:
    structural_feat: the structural feature to be normalized
    structural_feature_mean: the mean of the structural feature
    structural_feature_std: the standard deviation of the structural feature
  Returns:
    normalized_structural_feat: the normalized structural feature
  """
  return (
      structural_feat - torch.FloatTensor(structural_feature_mean)
  ) / torch.FloatTensor(structural_feature_std)


def train(
    model,
    data,
    train_loader,
    device,
    min_dst_idx,
    max_dst_idx,
    last_neighbor_loader,
    metrics_logger,
    structural_feats_list,
    structural_features,
):
  r"""Training procedure for TGN model

  This function uses some objects that are globally defined in the current
  scripts.

  Arguments:
      model: the model to be trained
      data: the dataset to be trained on
      train_loader: the loader for the training dataset with batches
      device: the device to run the training on
      min_dst_idx: the minimum destination index in the dataset
      max_dst_idx: the maximum destination index in the dataset
      last_neighbor_loader: the neighbor loader object
      use_xm: whether to use XManager
      xm_client: the XManager client
      logging_frequency: the frequency to log the training loss
      metrics_logger: the object to log the training metrics
      structural_feats_list: the list of structural features to be used
        (optional)
      structural_features: the structural features of the nodes as a dict
        (optional)

  Returns:
      None
  """

  model.initialize_train()
  if model.has_memory:
    model.reset_memory()  # Start with a fresh memory.
  last_neighbor_loader.reset_state()  # Start with an empty graph

  total_loss = 0
  curr_num_events = 0
  batch_id = 0
  with tqdm.tqdm(train_loader, unit='train_batch') as train_tqdm:
    for batch in train_tqdm:
      model.initialize_batch(batch)

      src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

      predicted_memory_embeddings = []
      original_memory_embeddings = []
      if model.has_struct_mapper:
        nodes = structural_features[batch_id][structural_feats_list[0]].keys()
        for node in nodes:
          different_feats = []
          for feat_type in structural_feats_list:
            if feat_type in structural_features[batch_id]:
              different_feats.append(
                  torch.Tensor(structural_features[batch_id][feat_type][node])
              )
          struct_feature_node_original = torch.cat(different_feats)
          struct_feature_node = normalize_structural_features(
              struct_feature_node_original,
              model.structural_feature_mean,
              model.structural_feature_std,
          )
          predicted_memory_embeddings.append(
              model.predict_memory_embeddings(struct_feature_node)
              .detach()
              .numpy()
          )
          original_memory_embeddings.append(
              model.get_memory_embeddings(torch.tensor([node])).detach().numpy()
          )
        batch_id += 1

      # Sample negative destination nodes.
      neg_dst = torch.randint(
          min_dst_idx,
          max_dst_idx + 1,
          (src.size(0),),
          dtype=torch.long,
          device=device,
      )

      model_prediction = model.predict_on_edges(
          source_nodes=src,
          target_nodes_pos=pos_dst,
          target_nodes_neg=neg_dst,
          last_neighbor_loader=last_neighbor_loader,
          data=data,
      )
      model_loss, structmap_loss = model.compute_loss(
          model_prediction,
          torch.tensor(np.array(predicted_memory_embeddings)),
          torch.tensor(np.array(original_memory_embeddings)),
      )
      if model.has_memory:
        model.update_memory(
            source_nodes=src,
            target_nodes_pos=pos_dst,
            target_nodes_neg=neg_dst,
            timestamps=t,
            messages=msg,
            last_neighbor_loader=last_neighbor_loader,
            data=data,
        )
      last_neighbor_loader.insert(src, pos_dst)
      loss = model_loss + structmap_loss
      model.optimize(loss)

      total_loss += float(loss) * batch.num_events

      curr_num_events += batch.num_events
      current_average_loss = total_loss / curr_num_events
      train_tqdm.set_postfix(avg_loss=current_average_loss)

      metrics_logger.global_train_steps += 1
      metrics_logger.global_total_loss += float(loss) * batch.num_events
      metrics_logger.global_num_events += batch.num_events


  return total_loss / data.num_events


@torch.no_grad()
def test(
    model,
    device,
    evaluator,
    last_neighbor_loader,
    data,
    metric,
    loader,
    neg_sampler,
    split_mode,
    update_memory,
    warmstart_batch_id,
    structural_feats_list,
    structural_features,
):
  r"""Evaluated the dynamic link prediction Evaluation happens as 'one vs.

  many', meaning that each positive edge is evaluated against many negative
  edges.

  Arguments:
      model: the model to be evaluated
      device: the device to run the evaluation on
      evaluator: the evaluator object
      last_neighbor_loader: the neighbor loader object
      data: the dataset to be evaluated on
      metric: the metric to be evaluated
      loader: an object containing positive attributes of the positive edges of
      the evaluation set
      neg_sampler: an object that gives the negative edges corresponding to each
      positive edge
      split_mode: specifies whether it is the 'validation' or 'test' set to
      correctly load the negatives
      update_memory: whether to update the memory while testing
      warmstart_batch_id: the batch id at the end of the warmstart set
      structural_feats_list: the list of structural features to be used
      (optional)
      structural_features: the structural features of the nodes as a dict
      (optional)
  Returns:
      perf_metric: the result of the performance evaluation as mrr
      auc: the result of the performance evaluation as auc
      PerformanceMetricLists: the performance metrics for each batch
  """
  model.initialize_test()

  perf_list = []
  lp = []  # Used for computing AUROC
  ln = []  # Used for computing AUROC
  loss_list_batch = []
  model_loss_list_batch = []
  perf_list_batch = []
  auc_list_batch = []
  track_nodes = set()
  num_steps = 0
  batch_id = warmstart_batch_id
  with tqdm.tqdm(loader, unit='test_batch') as test_tqdm:
    for pos_batch in test_tqdm:
      perf_sublist_batch = []
      lp_batch = []
      ln_batch = []
      batch_loss = 0
      batch_model_loss = 0
      pos_src, pos_dst, pos_t, pos_msg = (
          pos_batch.src,
          pos_batch.dst,
          pos_batch.t,
          pos_batch.msg,
      )

      predicted_memory_embeddings = []
      original_memory_embeddings = []
      if model.has_struct_mapper:
        nodes = set(
            structural_features[batch_id][structural_feats_list[0]].keys()
        )
        new_nodes = nodes - track_nodes
        track_nodes.update(nodes)
        for node in new_nodes:
          different_feats = []
          for feat_type in structural_feats_list:
            if feat_type in structural_features[batch_id]:
              different_feats.append(
                  torch.Tensor(structural_features[batch_id][feat_type][node])
              )
          struct_feature_node_original = torch.cat(different_feats)
          struct_feature_node = normalize_structural_features(
              struct_feature_node_original,
              model.structural_feature_mean,
              model.structural_feature_std,
          )
          model.initialize_memory_embedding(
              torch.tensor([node]),
              model.predict_memory_embeddings(struct_feature_node)[0],
          )
        for node in nodes:
          different_feats = []
          for feat_type in structural_feats_list:
            if feat_type in structural_features[batch_id]:
              different_feats.append(
                  torch.Tensor(structural_features[batch_id][feat_type][node])
              )
          struct_feature_node_original = torch.cat(different_feats)
          struct_feature_node = normalize_structural_features(
              struct_feature_node_original,
              model.structural_feature_mean,
              model.structural_feature_std,
          )
          predicted_memory_embeddings.append(
              model.predict_memory_embeddings(struct_feature_node)
              .detach()
              .numpy()
          )
          original_memory_embeddings.append(
              model.get_memory_embeddings(torch.tensor([node])).detach().numpy()
          )
        batch_id += 1

      neg_batch_list = neg_sampler.query_batch(
          pos_src, pos_dst, pos_t, split_mode=split_mode
      )

      for idx, neg_batch in enumerate(neg_batch_list):
        src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
        dst = torch.tensor(
            np.concatenate(
                ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                axis=0,
            ),
            device=device,
        )

        model_prediction = model.predict_on_edges(
            source_nodes=src,
            target_nodes_pos=dst,
            last_neighbor_loader=last_neighbor_loader,
            data=data,
        )
        model_loss, structmap_loss = model.compute_loss(
            model_prediction,
            torch.tensor(np.array(predicted_memory_embeddings)),
            torch.tensor(np.array(original_memory_embeddings)),
        )
        loss = model_loss + structmap_loss
        batch_loss += float(loss) * pos_batch.num_events
        batch_model_loss += float(model_loss) * pos_batch.num_events

        # compute MRR
        input_dict = {
            'y_pred_pos': np.array(
                [model_prediction.y_pred_pos[0, :].squeeze(dim=-1).cpu()]
            ),
            'y_pred_neg': np.array(
                model_prediction.y_pred_pos[1:, :].squeeze(dim=-1).cpu()
            ),
            'eval_metric': [metric],
        }
        perf_list.append(evaluator.eval(input_dict)[metric])
        perf_sublist_batch.append(evaluator.eval(input_dict)[metric])

        lp.extend(list(input_dict['y_pred_pos']))
        ln.extend(list(input_dict['y_pred_neg']))
        lp_batch.extend(list(input_dict['y_pred_pos']))
        ln_batch.extend(list(input_dict['y_pred_neg']))

      # Update memory and neighbor loader with ground-truth state.
      if update_memory and model.has_memory:
        model.update_memory(
            source_nodes=pos_src,
            target_nodes_pos=pos_dst,
            target_nodes_neg=None,
            timestamps=pos_t,
            messages=pos_msg,
            last_neighbor_loader=last_neighbor_loader,
            data=data,
        )
      last_neighbor_loader.insert(pos_src, pos_dst)
      num_steps += 1

      test_tqdm.set_postfix(perf_metric=float(torch.tensor(perf_list).mean()))

      loss_list_batch.append(batch_loss / len(neg_batch_list))
      model_loss_list_batch.append(batch_model_loss / len(neg_batch_list))
      perf_list_batch.append(np.mean(perf_sublist_batch))
      auc_list_batch.append(
          roc_auc_score(
              [1] * len(lp_batch) + [0] * len(ln_batch), lp_batch + ln_batch
          )
      )

  perf_metrics = float(torch.tensor(perf_list).mean())
  auc = roc_auc_score([1] * len(lp) + [0] * len(ln), lp + ln)

  return (
      perf_metrics,
      auc,
      PerformanceMetricLists(
          loss=loss_list_batch,
          model_loss=model_loss_list_batch,
          perf=perf_list_batch,
          auc=auc_list_batch,
      ),
  )


def get_warmstart_timestamp(initial_time_stamp, warmstart_days):

  u = datetime.fromtimestamp(initial_time_stamp)
  d = dtfull.timedelta(days=warmstart_days)
  t = u + d

  return int(datetime.timestamp(t))


def split_for_warmstart_days(test_data, warmstart_days):

  warmstart_upper = get_warmstart_timestamp(
      min(test_data.t.numpy()), warmstart_days
  )
  pretrain_mask = test_data.t < warmstart_upper

  return test_data[pretrain_mask], test_data[~pretrain_mask]


def split_for_warmstart_quantile(test_data, warmstart_quantile):

  pretrain_mask = test_data.t < torch.quantile(
      test_data.t.double(), warmstart_quantile
  )

  return test_data[pretrain_mask], test_data[~pretrain_mask]


def split_for_warmstart_batches(test_data, batch_fraction, batch_size):

  number_of_batches = np.ceil(len(test_data) / batch_size)
  batches_in_warmstart = np.ceil(number_of_batches * batch_fraction)

  return (
      test_data[0 : int(batch_size * batches_in_warmstart)],
      test_data[int(batch_size * batches_in_warmstart) :],
      batches_in_warmstart,
  )


def warmstart(
    model,
    data,
    data_loader,
    device,
    min_dst_idx,
    max_dst_idx,
    metric,
    last_neighbor_loader,
    evaluator,
    metrics_logger,
    update_model,
    structural_feats_list,
    structural_features,
):
  r"""Warmstart on a fraction of test dataset.

  This function uses some objects that are globally defined in the current
  scripts.

  Arguments:
      model: the model to be evaluated
      data: the dataset to be evaluated on
      data_loader: an object containing positive attributes of the positive
        edges of the evaluation set
      device: accelerator, if used
      min_dst_idx: the minimum destination index to be used for negative
        sampling
      max_dst_idx: the maximum destination index to be used for negative
        sampling
      metric: the metric to be evaluated
      last_neighbor_loader: the neighbor loader object
      evaluator: the evaluator object
      metrics_logger: the object to log the training metrics
      structural_feats_list: the list of structural features to be used
        (optional)
      structural_features: the structural features of the nodes as a dict
        (optional)

  Returns:
      PerformanceMetricLists: the performance metrics for each batch
  """

  if update_model:
    model.initialize_train()
  else:
    model.initialize_test()

  loss_list_batch = []
  model_loss_list_batch = []
  perf_list_batch = []
  auc_list_batch = []
  total_loss = 0
  curr_num_events = 0
  track_nodes = set()
  batch_id = 0
  with tqdm.tqdm(data_loader, unit='train_batch') as train_tqdm:
    for batch in train_tqdm:
      model.initialize_batch(batch)

      src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

      predicted_memory_embeddings = []
      original_memory_embeddings = []
      if model.has_struct_mapper:
        nodes = set(
            structural_features[batch_id][structural_feats_list[0]].keys()
        )
        new_nodes = nodes - track_nodes
        track_nodes.update(nodes)
        for node in new_nodes:
          different_feats = []
          for feat_type in structural_feats_list:
            if feat_type in structural_features[batch_id]:
              different_feats.append(
                  torch.Tensor(structural_features[batch_id][feat_type][node])
              )
          struct_feature_node_original = torch.cat(different_feats)
          struct_feature_node = normalize_structural_features(
              struct_feature_node_original,
              model.structural_feature_mean,
              model.structural_feature_std,
          )
          model.initialize_memory_embedding(
              torch.tensor([node]),
              model.predict_memory_embeddings(struct_feature_node)[0],
          )
        for node in nodes:
          different_feats = []
          for feat_type in structural_feats_list:
            if feat_type in structural_features[batch_id]:
              different_feats.append(
                  torch.Tensor(structural_features[batch_id][feat_type][node])
              )
          struct_feature_node_original = torch.cat(different_feats)
          struct_feature_node = normalize_structural_features(
              struct_feature_node_original,
              model.structural_feature_mean,
              model.structural_feature_std,
          )
          predicted_memory_embeddings.append(
              model.predict_memory_embeddings(struct_feature_node)
              .detach()
              .numpy()
          )
          original_memory_embeddings.append(
              model.get_memory_embeddings(torch.tensor([node])).detach().numpy()
          )
        batch_id += 1

      # Sample negative destination nodes.
      neg_dst = torch.randint(
          min_dst_idx,
          max_dst_idx + 1,
          (src.size(0),),
          dtype=torch.long,
          device=device,
      )

      if model.has_memory:
        model.update_memory(
            source_nodes=src,
            target_nodes_pos=pos_dst,
            target_nodes_neg=neg_dst,
            timestamps=t,
            messages=msg,
            last_neighbor_loader=last_neighbor_loader,
            data=data,
        )
      last_neighbor_loader.insert(src, pos_dst)

      model_prediction = model.predict_on_edges(
          source_nodes=src,
          target_nodes_pos=pos_dst,
          target_nodes_neg=neg_dst,
          last_neighbor_loader=last_neighbor_loader,
          data=data,
      )
      model_loss, structmap_loss = model.compute_loss(
          model_prediction,
          torch.tensor(np.array(predicted_memory_embeddings)),
          torch.tensor(np.array(original_memory_embeddings)),
      )
      loss = model_loss + structmap_loss
      total_loss += float(loss) * batch.num_events
      loss_list_batch.append(float(loss) * batch.num_events)
      model_loss_list_batch.append(float(model_loss) * batch.num_events)

      # compute MRR
      input_dict = {
          'y_pred_pos': np.array([
              model_prediction.y_pred_pos[0, :]
              .squeeze(dim=-1)
              .cpu()
              .detach()
              .numpy()
          ]),
          'y_pred_neg': np.array(
              model_prediction.y_pred_pos[1:, :]
              .squeeze(dim=-1)
              .cpu()
              .detach()
              .numpy()
          ),
          'eval_metric': [metric],
      }

      lp_batch = list(input_dict['y_pred_pos'])
      ln_batch = list(input_dict['y_pred_neg'])
      perf_list_batch.append(evaluator.eval(input_dict)[metric])
      auc_list_batch.append(
          roc_auc_score(
              [1] * len(lp_batch) + [0] * len(ln_batch), lp_batch + ln_batch
          )
      )

      if update_model:
        model.optimize(loss)
        curr_num_events += batch.num_events
        current_average_loss = total_loss / curr_num_events
        train_tqdm.set_postfix(avg_loss=current_average_loss)

        metrics_logger.global_train_steps += 1
        metrics_logger.global_total_loss += float(loss) * batch.num_events
        metrics_logger.global_num_events += batch.num_events


  return PerformanceMetricLists(
      loss=loss_list_batch,
      model_loss=model_loss_list_batch,
      perf=perf_list_batch,
      auc=auc_list_batch,
  )
