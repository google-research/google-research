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

"""TFDS input pipelines for GLUE and C4."""

import jax
import numpy as np
import tensorflow_datasets as tfds


def tfds_stream(dataset_name, split, batch_size, data_dir,
                shuffle_files, shuffle_buffer_size, batch_shuffle_size,
                preprocess_fun, repeat=True):
  """Streams batches of examples from tfds, with pure-python preprocessing."""
  ds = tfds.load(
      name=dataset_name, split=split, data_dir=data_dir,
      shuffle_files=shuffle_files)
  if repeat:
    ds = ds.cache()
    ds = ds.repeat()
  if shuffle_buffer_size is not None:
    ds = ds.shuffle(shuffle_buffer_size)
  ds = ds.batch(batch_size)
  if batch_shuffle_size is not None:
    ds = ds.shuffle(batch_shuffle_size)

  for batch in tfds.as_numpy(ds):
    if preprocess_fun is not None:
      yield preprocess_fun(batch)
    else:
      yield batch


def glue_inputs(dataset_name, split, batch_size, tokenizer, data_dir=None,
                max_len=128, training=True):
  """Input pipeline for fine-tuning BERT on GLUE tasks."""
  keys_lookup = {
      "glue/cola": ("sentence", None),
      "glue/sst2": ("sentence", None),
      "glue/mrpc": ("sentence1", "sentence2"),
      "glue/qqp": ("question1", "question2"),
      "glue/stsb": ("sentence1", "sentence2"),
      "glue/mnli": ("premise", "hypothesis"),   # TODO(kitaev): swap the two?
      "glue/qnli": ("question", "sentence"),  # TODO(kitaev) swap the two?
      "glue/rte": ("sentence1", "sentence2"),
      "glue/wnli": ("sentence1", "sentence2"),
  }

  key_a, key_b = keys_lookup[dataset_name]

  if key_b is None:
    def preprocess(batch):
      """Tokenize and convert text to model inputs."""
      batch_size = batch["idx"].shape[0]
      input_ids = np.zeros((batch_size, max_len), dtype=np.int32)
      type_ids = np.zeros((batch_size, max_len), dtype=np.int32)

      for i in range(batch_size):
        sentence_a = batch[key_a][i]
        tokens_a = tokenizer.EncodeAsIds(sentence_a)
        input_ids[i, :len(tokens_a)] = tokens_a[:max_len]

      return {
          "input_ids": input_ids,
          "type_ids": type_ids,
          "idx": batch["idx"].astype(np.int32),
          "label": batch["label"],
      }
  else:
    def preprocess(batch):
      """Tokenize and convert text to model inputs."""
      batch_size = batch["idx"].shape[0]
      input_ids = np.zeros((batch_size, max_len), dtype=np.int32)
      type_ids = np.zeros((batch_size, max_len), dtype=np.int32)

      for i in range(batch_size):
        sentence_a = batch[key_a][i]
        sentence_b = batch[key_b][i]
        tokens_a = tokenizer.EncodeAsIds(sentence_a)
        tokens_b = tokenizer.EncodeAsIds(sentence_b)[1:]  # Strip start token

        ex_input_ids = (tokens_a + tokens_b)[:max_len]
        ex_type_ids = ([0] * len(tokens_a) + [1] * len(tokens_b))[:max_len]

        input_ids[i, :len(ex_input_ids)] = ex_input_ids
        type_ids[i, :len(ex_type_ids)] = ex_type_ids

      return {
          "input_ids": input_ids,
          "type_ids": type_ids,
          "idx": batch["idx"].astype(np.int32),
          "label": batch["label"],
      }

  return tfds_stream(
      dataset_name=dataset_name,
      split=split,
      batch_size=batch_size,
      data_dir=data_dir,
      shuffle_files=training,
      shuffle_buffer_size=1024 if training else None,
      batch_shuffle_size=128 if training else None,
      preprocess_fun=preprocess,
      repeat=training,
      )


def _c4_data_unbatched(tokenizer, max_len):
  """Yields examples from the C4 corpus that have len(text) <= max_len."""
  cls_id = tokenizer.bos_id()
  sep_id = tokenizer.eos_id()
  pad_id = tokenizer.pad_id()

  ds = tfds.load(name="c4/en", split="train", shuffle_files=True)
  ds = ds.repeat()
  ds = ds.shuffle(1024)
  ds = ds.batch(16)  # Batch documents to potentially speed up input pipeline

  input_ids_buf = np.full((1024, max_len), pad_id, dtype=np.int32)
  type_ids_buf = np.full((1024, max_len), pad_id, dtype=np.int32)
  next_sentence_labels_buf = np.full(1024, -1, dtype=np.int32)

  for batch in tfds.as_numpy(ds):
    for text in batch["text"]:
      text = str(text, "utf-8")
      lines = [tokenizer.EncodeAsIds(line) for line in text.splitlines()]
      j = 0
      while j < len(lines) - 1:
        if len(lines[j]) + len(lines[j+1]) > max_len - 3:
          j += 1
        else:
          idx = np.random.randint(input_ids_buf.shape[0])
          if next_sentence_labels_buf[idx] != -1:
            yield {
                "input_ids": input_ids_buf[idx].copy(),
                "type_ids": type_ids_buf[idx].copy(),
                "next_sentence_labels": next_sentence_labels_buf[idx].copy(),
            }
          input_ids_buf[idx] = pad_id
          type_ids_buf[idx] = 1

          cum_len = 0
          for k in range(j, len(lines)):
            cum_len += len(lines[k])
            if cum_len > max_len - 3:
              k -= 1
              break
          selected_lines = lines[j:k+1]
          j = k + 1

          pivot = np.random.randint(1, len(selected_lines))
          if np.random.random() < 0.5:
            datum = [cls_id]
            for tokens in selected_lines[:pivot]:
              datum.extend(tokens)
            datum.append(sep_id)
            type_ids_buf[idx, :len(datum)] = 0
            for tokens in selected_lines[pivot:]:
              datum.extend(tokens)
            datum.append(sep_id)
            next_sentence_label = 0
            type_ids_buf[idx, len(datum):] = 0
          else:
            datum = [cls_id]
            for tokens in selected_lines[pivot:]:
              datum.extend(tokens)
            datum.append(sep_id)
            type_ids_buf[idx, :len(datum)] = 0
            for tokens in selected_lines[:pivot]:
              datum.extend(tokens)
            datum.append(sep_id)
            next_sentence_label = 1
            type_ids_buf[idx, len(datum):] = 0

          input_ids_buf[idx] = pad_id
          input_ids_buf[idx, :len(datum)] = datum
          next_sentence_labels_buf[idx] = next_sentence_label


def c4_masked_lm_inputs(batch_size, tokenizer, max_len,
                        max_predictions_per_seq):
  """"Generates a batch of masked examples from the C4 corpus."""
  ignore_ids = [tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()]
  ignore_ids = np.array(ignore_ids, dtype=np.int32)[:, None]
  pad_id = tokenizer.pad_id()
  mask_id = tokenizer.PieceToId("[MASK]")

  it = _c4_data_unbatched(tokenizer, max_len)
  examples = []
  while True:
    example = next(it)

    num_tokens = np.sum(example["input_ids"] != pad_id).item()
    prediction_mask = np.all(example["input_ids"] != ignore_ids, axis=0)
    cand_indexes = np.arange(
        prediction_mask.shape[0], dtype=np.int32)[prediction_mask]
    num_to_predict = min(
        max_predictions_per_seq, max(1, int(num_tokens * 0.15)))

    masked_lm_positions = np.random.choice(
        cand_indexes, num_to_predict, replace=False)
    masked_lm_positions = np.sort(masked_lm_positions)
    masked_lm_ids = example["input_ids"][masked_lm_positions]
    example["input_ids"][masked_lm_positions] = mask_id
    masked_lm_weights = np.ones_like(masked_lm_positions, dtype=np.float32)

    amount_to_pad = max_predictions_per_seq - num_to_predict
    masked_lm_positions = np.pad(
        masked_lm_positions, (0, amount_to_pad), mode="constant")
    masked_lm_ids = np.pad(
        masked_lm_ids, (0, amount_to_pad), mode="constant")
    masked_lm_weights = np.pad(
        masked_lm_weights, (0, amount_to_pad), mode="constant")

    example["masked_lm_positions"] = masked_lm_positions
    example["masked_lm_ids"] = masked_lm_ids
    example["masked_lm_weights"] = masked_lm_weights

    examples.append(example)
    if len(examples) == batch_size:
      yield jax.tree_multimap(lambda *x: np.stack(x), *examples)
      examples = []
