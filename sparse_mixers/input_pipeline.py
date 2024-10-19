# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""TFDS input pipelines for GLUE, SuperGLUE and C4 datasets."""

import re
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import sentencepiece as spm

NumpyBatch = Dict[str, np.ndarray]
NumpyExample = Dict[str, Any]


def _tfds_stream(dataset_name,
                 split,
                 batch_size,
                 data_dir,
                 shuffle_files,
                 shuffle_buffer_size,
                 batch_shuffle_size,
                 postprocess_fn,
                 preprocess_fn = None,
                 repeat = True):
  """Streams batches of examples from TFDS, with pure-python pre-processing."""
  ds = tfds.load(
      name=dataset_name,
      split=split,
      data_dir=data_dir,
      shuffle_files=shuffle_files)

  if preprocess_fn is not None:
    ds_numpy = list(tfds.as_numpy(ds))
    ds = tf.data.Dataset.from_tensor_slices(preprocess_fn(ds_numpy))

  if repeat:
    ds = ds.cache()
    ds = ds.repeat()
  if shuffle_buffer_size is not None:
    ds = ds.shuffle(shuffle_buffer_size)

  ds = ds.batch(batch_size)

  if batch_shuffle_size is not None:
    ds = ds.shuffle(batch_shuffle_size)

  for batch in tfds.as_numpy(ds):
    result = postprocess_fn(batch)
    if result:
      yield result


def classification_inputs(dataset_name,
                          split,
                          batch_size,
                          tokenizer,
                          data_dir = None,
                          max_seq_length = 128,
                          training = True):
  """Input pipeline for fine-tuning on classification tasks.

  Args:
    dataset_name: TFDS dataset name.
    split: Which dataset split to use (TRAINING, TEST or VALIDATION)
    batch_size: Number of examples in each batch.
    tokenizer: Tokenizer for converting text to integers representations.
    data_dir: Optional directory from which to load dataset.
    max_seq_length: Sequences longer than this are truncated; shorter sequences
      are padded.
    training: In training mode, we shuffle, repeat and buffer the dataset.

  Returns:
    Batched examples for specified dataset with keys and array types/shapes:
      * "input_ids": <np.int32>[batch_size, max_seq_length]
      * "type_ids": <np.int32>[batch_size, max_seq_length]
      * "idx": <np.int32>[batch_size]
      * "label": <np.int32>[batch_size, NUM_LABELS]
  """
  keys_lookup = {
      "glue/cola": ("sentence",),
      "glue/sst2": ("sentence",),
      "glue/mrpc": ("sentence1", "sentence2"),
      "glue/qqp": ("question1", "question2"),
      "glue/stsb": ("sentence1", "sentence2"),
      "glue/mnli": ("hypothesis", "premise"),
      "glue/qnli": ("question", "sentence"),
      "glue/rte": ("sentence1", "sentence2"),
      # WNLI requires a special training recipe. Following the original BERT
      # paper, we don't eval on it.
      "glue/wnli": ("sentence1", "sentence2"),
      "super_glue/boolq": ("question", "passage"),
      "super_glue/cb": ("hypothesis", "premise"),
      # For COPA, see _singularize_copa_examples() for pre-processing.
      "super_glue/copa": ("premise", "question", "choice"),
      "super_glue/multirc": ("question", "answer", "paragraph"),
      # For ReCoRD, see _singularize_record_examples() for pre-processing.
      "super_glue/record": ("entity", "query", "passage"),
      "super_glue/rte": ("hypothesis", "premise"),
      "super_glue/wic": ("word", "sentence1", "sentence2"),
      # Like WNLI, WSC  is a span-based task which requires a special setup, so
      # we don't eval on it.
      "super_glue/wsc": ("span1_text", "span2_text", "text"),
      "super_glue/wsc.fixed": ("span1_text", "span2_text", "text"),
      # AXB and AXG are diagnostic tasks, which we don't eval on.
      "super_glue/axb": ("sentence1", "sentence2"),
      "super_glue/axg": ("hypothesis", "premise"),
  }

  cls_id = tokenizer.PieceToId("[CLS]")
  sep_id = tokenizer.PieceToId("[SEP]")
  pad_id = tokenizer.pad_id()

  def process_classification(batch):
    """Tokenizes and converts GLUE/SuperGLUE examples to model inputs."""
    keys = keys_lookup[dataset_name]

    if dataset_name == "super_glue/multirc":
      idx = batch["idx"]["question"]
    else:
      idx = batch["idx"]
    num_examples = idx.shape[0]

    input_ids = np.full((num_examples, max_seq_length), pad_id, dtype=np.int32)
    type_ids = np.zeros((num_examples, max_seq_length), dtype=np.int32)

    for i in range(num_examples):
      ex_input_ids = [cls_id]
      ex_type_ids = [0]
      for type_id, key in enumerate(keys):
        inputs = _clean_multirc_inputs(dataset_name, batch[key][i])
        tokens = tokenizer.EncodeAsIds(inputs) + [sep_id]
        ex_input_ids.extend(tokens)
        ex_type_ids.extend([type_id] * len(tokens))

      ex_input_ids = ex_input_ids[:max_seq_length]
      ex_type_ids = ex_type_ids[:max_seq_length]
      input_ids[i, :len(ex_input_ids)] = ex_input_ids
      type_ids[i, :len(ex_type_ids)] = ex_type_ids

    return {
        "input_ids": input_ids,
        "type_ids": type_ids,
        "idx": idx.astype(np.int32),
        "label": batch["label"]
    }

  # Convert multi-answer examples into single-answer examples.
  if dataset_name == "super_glue/record":
    preprocess_fn = _singularize_record_examples
  elif dataset_name == "super_glue/copa":
    preprocess_fn = _singularize_copa_examples
  else:
    preprocess_fn = None

  return _tfds_stream(
      dataset_name=dataset_name,
      split=split,
      batch_size=batch_size,
      data_dir=data_dir,
      shuffle_files=training,
      shuffle_buffer_size=1024 if training else None,
      batch_shuffle_size=128 if training else None,
      preprocess_fn=preprocess_fn,
      postprocess_fn=process_classification,
      repeat=training)


def _clean_multirc_inputs(dataset_name, text):
  """Removes HTML markup from Multi-RC task input text."""
  if dataset_name == "super_glue/multirc":
    # Remove HTML markup.
    text = re.sub(r"<br>", " ", text.decode("utf-8"))
    text = re.sub(r"<(/)?b>", " ", text)
  return text


def _singularize_copa_examples(dataset):
  """Converts COPA multi-choice examples to batch of single-answer examples."""
  indexes = []
  premises = []
  questions = []
  choices = []
  labels = []

  for example in dataset:
    for choice in ["choice1", "choice2"]:
      indexes.append(example["idx"])
      premises.append(example["premise"])
      questions.append(example["question"])
      choices.append(example[choice])
      if choice == "choice1":
        label = example["label"] == 0
      else:
        label = example["label"] == 1
      labels.append(label)

  return {
      "idx": np.array(indexes),
      "premise": np.array(premises),
      "question": np.array(questions),
      "choice": np.array(choices),
      "label": np.array(labels)
  }


def _singularize_record_examples(dataset):
  """Converts ReCoRD multi-answer examples to single-answer batch."""
  indexes = []
  passages = []
  entities = []
  queries = []
  labels = []

  for example in dataset:
    for entity in example["entities"]:
      indexes.append(example["idx"]["query"])
      passages.append(example["passage"])
      queries.append(example["query"])
      entities.append(entity)
      labels.append(entity in example["answers"])

  return {
      "idx": np.array(indexes),
      "passage": np.array(passages),
      "query": np.array(queries),
      "entity": np.array(entities),
      "label": np.array(labels)
  }


def _c4_data_unbatched(tokenizer,
                       max_seq_length):
  """Yields examples from C4 corpus that have len(text) <= max_seq_length."""
  cls_id = tokenizer.PieceToId("[CLS]")
  sep_id = tokenizer.PieceToId("[SEP]")
  pad_id = tokenizer.pad_id()

  ds = tfds.load(name="c4/en", split="train", shuffle_files=True)
  ds = ds.repeat()
  ds = ds.shuffle(1024)
  ds = ds.shuffle(1024)
  ds = ds.batch(16)  # Batch documents to potentially speed up input pipeline

  input_ids_buf = np.full((1024, max_seq_length), pad_id, dtype=np.int32)
  type_ids_buf = np.full((1024, max_seq_length), pad_id, dtype=np.int32)
  next_sentence_labels_buf = np.full(1024, -1, dtype=np.int32)

  for batch in tfds.as_numpy(ds):
    for text in batch["text"]:
      text = str(text, "utf-8")
      lines = [tokenizer.EncodeAsIds(line) for line in text.splitlines()]
      j = 0
      while j < len(lines) - 1:
        if len(lines[j]) + len(lines[j + 1]) > max_seq_length - 3:
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
            if cum_len > max_seq_length - 3:
              k -= 1
              break
          selected_lines = lines[j:k + 1]
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
            next_sentence_label = 1
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
            next_sentence_label = 0
            type_ids_buf[idx, len(datum):] = 0

          input_ids_buf[idx] = pad_id
          input_ids_buf[idx, :len(datum)] = datum
          next_sentence_labels_buf[idx] = next_sentence_label


def c4_masked_lm_inputs(batch_size, tokenizer,
                        max_seq_length, max_predictions_per_seq,
                        masking_rate, mask_token_proportion,
                        random_token_proportion):
  """"Generates a batch of masked examples from the C4 corpus.

  Args:
    batch_size: Number of examples in each batch.
    tokenizer: Tokenizer for converting text to integers representations.
    max_seq_length: Sequences longer than this are truncated; shorter sequences
      are padded.
    max_predictions_per_seq: Maximum number of masked LM predictions per
      sequence.
    masking_rate: Proportion of tokens for masked LM predictions. Total number
      of selected tokens will be at most max_predictions_per_seq.
    mask_token_proportion: Proportion of masked tokens to replace with ['MASK'].
    random_token_proportion: Proportion of masked tokens to replace with a
      random token. Remaining 1-mask_token_proportion-random_token_proportion
      fraction of selected tokens are left as is.

  Yields:
    Batches of examples with keys and array types/shapes:
    * "input_ids": <np.int32>[batch_size, max_seq_length]
    * "type_ids": <np.int32>[batch_size, max_seq_length]
    * "masked_lm_positions": <np.int32>[batch_size, max_predictions_per_seq]
    * "masked_lm_ids": <np.int32>[batch_size ,max_predictions_per_seq]
    * "masked_lm_weights": <np.int32>[batch_size, max_predictions_per_seq]
    * "next_sentence_labels": <np.int32>[batch_size]
  """
  total = mask_token_proportion + random_token_proportion
  if total < 0 or total > 1:
    raise ValueError(
        "Sum of random proportion and mask proportion must be in [0, 1] range. "
        "Got random_token_proportion=%d and mask_token_proportion=%d" %
        (random_token_proportion, mask_token_proportion))

  pad_id = tokenizer.pad_id()
  eos_id = tokenizer.eos_id()
  bos_id = tokenizer.bos_id()
  cls_id = tokenizer.PieceToId("[CLS]")
  sep_id = tokenizer.PieceToId("[SEP]")
  mask_id = tokenizer.PieceToId("[MASK]")
  ignore_ids = [cls_id, sep_id, pad_id]
  ignore_ids = np.array(ignore_ids, dtype=np.int32)[:, None]

  special_tokens = {mask_id, cls_id, sep_id, bos_id, eos_id, pad_id}
  normal_tokens = [
      t for t in range(tokenizer.GetPieceSize()) if t not in special_tokens
  ]

  it = _c4_data_unbatched(tokenizer, max_seq_length)
  examples = []
  while True:
    example = next(it)

    num_tokens = np.sum(example["input_ids"] != pad_id).item()
    prediction_mask = np.all(example["input_ids"] != ignore_ids, axis=0)
    cand_indexes = np.arange(
        prediction_mask.shape[0], dtype=np.int32)[prediction_mask]
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(num_tokens * masking_rate)))

    masked_lm_positions = np.random.choice(
        cand_indexes, num_to_predict, replace=False)
    masked_lm_positions = np.sort(masked_lm_positions)
    masked_lm_ids = example["input_ids"][masked_lm_positions]
    masked_lm_weights = np.ones_like(masked_lm_positions, dtype=np.float32)

    # Mask out tokens.
    for position in masked_lm_positions:
      rand = np.random.random()
      if rand < mask_token_proportion:
        replace_token_id = mask_id
      elif rand < mask_token_proportion + random_token_proportion:
        replace_token_id = np.random.choice(normal_tokens, 1).item()
      else:
        replace_token_id = example["input_ids"][position]
      example["input_ids"][position] = replace_token_id

    amount_to_pad = max_predictions_per_seq - num_to_predict
    masked_lm_positions = np.pad(
        masked_lm_positions, (0, amount_to_pad), mode="constant")
    masked_lm_ids = np.pad(masked_lm_ids, (0, amount_to_pad), mode="constant")
    masked_lm_weights = np.pad(
        masked_lm_weights, (0, amount_to_pad), mode="constant")

    example["masked_lm_positions"] = masked_lm_positions
    example["masked_lm_ids"] = masked_lm_ids
    example["masked_lm_weights"] = masked_lm_weights

    examples.append(example)
    if len(examples) == batch_size:
      yield jax.tree.map(lambda *x: np.stack(x), *examples)
      examples = []
