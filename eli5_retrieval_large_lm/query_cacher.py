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

r"""Iterates over the whole ELI5 dataset (for each dataset split), extracts REALM or REALM++ embeddings, then does exact retrieval for a large number of neighbors, then saves the resulting db indices and distance (inner product).

This makes it so we don't have to do real retrieval during training, we just
sample from the neighbors as a function of the inner product, which is much
faster.

Examples of use:

# Local Test
pytype query_cacher.py -P . --check-variable-types --check-container-types \
--check-parameter-types --precise-return && \
python3 check_flags.py query_cacher.py && \
python3 query_cacher.py $(python3 json_to_args.py
configs/query_cacher_configs/local.json) \
--logger_levels=__main__:DEBUG,utils:DEBUG,tf_utils:DEBUG \
--use_subset=True \
--do_retrieval=True

For remote use see google/xcloud_launch_query_cached.py

"""
import logging
import operator
import os
import pathlib
import subprocess
import tempfile
import time
from typing import List, Callable, Dict

from absl import app
from absl import flags
from absl import logging as absl_logging
import bert_utils
import colorama
import constants
import datasets
import h5py
import numpy as np
import tensorflow as tf
import tensorflow.python.distribute.values as values
import tensorflow.python.framework.ops as ops
import tensorflow.python.training.tracking.tracking as tracking
import tensorflow_hub as hub
import tf_utils
import tqdm
import transformers
import utils


_FLAG_SUBSET_AMOUNT = flags.DEFINE_integer(
    "subset_amount",
    100,
    "Quantity of samples to use in the subset."
)

_FLAG_JOB_NAME = flags.DEFINE_string(
    "run_name",
    None,
    "Name of the run."
)
_FLAG_OUTPUT_PATH = flags.DEFINE_string(
    "output_dir",
    None,
    "Directory in which to save, on the cloud.")
_FLAG_RETRIEVER_CONFIG_PATH = flags.DEFINE_string(
    "retriever_config_path",
    None,
    "Path to the retriever's configuration file."
)
_FLAG_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    100,
    "Size of the batch for the encoder BERT model."
)
_FLAG_USE_SUBSET = flags.DEFINE_boolean(
    "use_subset",
    False,
    "Whether to only use a small subset of the data or not."
)

_FLAGS_DATASET_ROOT = flags.DEFINE_string(
    "dataset_root",
    None,
    "Root of the place where the datasets are saved."
)

# Flags specific to query encoding
_FLAG_EMBEDDING_DEPTH = flags.DEFINE_integer(
    "embedding_depth",
    128,
    "Size of the BERT (REALM) embeddings.",
)

# Flags specific to retrieval caching
_FLAG_DO_RETRIEVAL = flags.DEFINE_boolean(
    "do_retrieval",
    True,
    "Whether to do the retrievals or to just save the encoded embeddings.",
)
_FLAG_NUM_RETRIEVALS = flags.DEFINE_integer(
    "num_retrievals",
    10,
    "Number of neighbors to retrieve.",
)
_FLAG_CONTEXT_SIZE = flags.DEFINE_integer(
    "context_size",
    1024,
    "Length to pad to."
)
_FLAG_MAX_LENGTH_RETRIEVALS = flags.DEFINE_integer(
    "max_length_retrievals",
    350,
    "Maximum length of the retrievals."
)

LOGGER = logging.getLogger(__name__)


class _BERTBatchFields(utils.FlagChoices):
  bert_question_token_ids = "bert_question_token_ids"
  bert_attention_masks = "bert_attention_masks"


def _warn_subset(tqdm_obj=None):
  """Warn that we are using a subset of the dataset."""
  message = (f"{colorama.Back.RED}{colorama.Fore.WHITE}"
             f"{colorama.Style.BRIGHT} >> USING A SUBSET OF THE DATASET << "
             f"{colorama.Style.RESET_ALL}")
  if tqdm_obj:
    tqdm_obj.write(message)
  else:
    LOGGER.warning(message)


def _make_transform_fn(
    bert_tokenizer,
    bert_cls_token_id,
    bert_sep_token_id,
):
  """Prepares the transformation function."""
  @tf.function
  def _prepare_for_bert(sample):
    """Prepares a question sample from ELI5 to be fed to BERT."""
    bert_question_token_ids = bert_tokenizer.tokenize(
        tf.expand_dims(sample["question"], 0))
    bert_question_token_ids = tf.cast(
        bert_question_token_ids.merge_dims(1, 2).to_tensor(), tf.int32)
    cls_ids = tf.fill([tf.shape(bert_question_token_ids)[0], 1],
                      bert_cls_token_id)
    sep_ids = tf.fill([tf.shape(bert_question_token_ids)[0], 1],
                      bert_sep_token_id)
    bert_question_token_ids = tf.concat(
        (cls_ids, bert_question_token_ids, sep_ids), 1)

    return dict(
        bert_question_token_ids=bert_question_token_ids,
        bert_attention_masks=tf.ones_like(bert_question_token_ids),
        **sample
    )

  return _prepare_for_bert


@tf.function
def _squeeze(batch):
  """Squeezes and converts tensors to dense tensors w/ padding."""
  batch = dict(**batch)
  batch[_BERTBatchFields.bert_question_token_ids] = tf.squeeze(
      batch[_BERTBatchFields.bert_question_token_ids].to_tensor(0), 1)
  batch[_BERTBatchFields.bert_attention_masks] = tf.squeeze(
      batch[_BERTBatchFields.bert_attention_masks].to_tensor(0), 1)
  return batch


def _make_encode_fn(
    query_encoder
):
  """Prepares the BERT encoder function."""

  @tf.function(experimental_relax_shapes=True)
  def _encode(batch):
    """Encodes a sample with REALM BERT."""
    # Add a CLS token at the start of the input, and a SEP token at the end

    return query_encoder.signatures["projected"](
        input_ids=batch[_BERTBatchFields.bert_question_token_ids],
        input_mask=batch[_BERTBatchFields.bert_attention_masks],
        segment_ids=tf.zeros_like(
            batch[_BERTBatchFields.bert_question_token_ids]
        ))["default"]

  return _encode


def _make_encode_fn_strategy_run_fn(
    strategy,
    encode_fn,
):
  """Builds the runner for the query encoder function."""

  # Giving {} as a default value would make the default value mutable, which
  # is prohibited (because changing the object would change the default value).

  @tf.function(experimental_relax_shapes=True)
  def encode_fn_strategy_run_fn(batch):
    """Runner for the query encoder function."""
    return strategy.run(encode_fn, args=(batch,))

  return encode_fn_strategy_run_fn


######################################################################
# Effectuate the retrievals.
######################################################################
def _prep_field(field, gpt2_tokenizer):
  """Prepares the different fields that are to be saved."""
  decoded_list = [sample.decode() for sample in field.numpy().tolist()]
  encoded = gpt2_tokenizer.batch_encode_plus(
      decoded_list,
      padding="max_length",
      truncation=True,
  ).input_ids

  ids = np.array(
      encoded,
      dtype=np.int32,
  )

  ids[ids == gpt2_tokenizer.eos_token_id] = -1
  return ids


def main(argv):
  if len(argv) > 1:
    raise RuntimeError(argv)
  absl_logging.use_python_logging()
  retriever_config = tf_utils.REALMSave(
      **utils.from_json_file(_FLAG_RETRIEVER_CONFIG_PATH.value))

  extra = "_FROM_SUBSET" if _FLAG_USE_SUBSET.value else ""
  time_stamp = time.strftime("%Y%m%d-%H%M%S")
  target_path = os.path.join(_FLAG_OUTPUT_PATH.value,
                             time_stamp + extra).strip()
  if target_path[-1] != "/":
    target_path += "/"

  ##############################################################################
  # Setup devices and strategy
  ##############################################################################
  with utils.log_duration(LOGGER, "main", "Initializing devices"):
    tpu_config = tf_utils.init_tpus()
    device_type = tf_utils.current_accelerator_type()
    LOGGER.debug("Devices: %s", str(tf_utils.devices_to_use()))

    if device_type == "TPU":
      if tpu_config is None:
        raise RuntimeError("We should have a tpu_config.")
      strategy = tf.distribute.TPUStrategy(tpu_config.resolver)
      batch_size = len(tf_utils.devices_to_use()) * _FLAG_BATCH_SIZE.value
    elif device_type == "GPU" or device_type == "CPU":
      strategy = tf.distribute.MirroredStrategy()
      batch_size = len(tf_utils.devices_to_use()) * _FLAG_BATCH_SIZE.value
    else:
      raise RuntimeError(device_type)

  ##############################################################################
  # Load the dataset.
  ##############################################################################
  eli5 = {}
  keys = ["train", "eval", "test"]
  gpt2_tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-xl")
  gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

  with utils.log_duration(LOGGER, "main", "Loading the ELI5 datasets."):
    for split in tqdm.tqdm(keys):
      load_path = os.path.join(
          _FLAGS_DATASET_ROOT.value,
          "HuggingfaceDatasets",
          f"{split}_kilt_eli5.hf"
      )
      with tf.device("/job:localhost"):
        eli5[split] = datasets.load_from_disk(load_path)

  if _FLAG_USE_SUBSET.value:
    _warn_subset()

  ##############################################################################
  #
  ##############################################################################
  with utils.log_duration(
      LOGGER, "Main", "Load the textual dataset"
  ):
    # Extract the appropriate text
    # The buffer_size is taken from the original ORQA code.
    blocks_dataset = tf.data.TFRecordDataset(
        retriever_config.text_records, buffer_size=512 * 1024 * 1024
    )
    blocks_dataset = blocks_dataset.batch(
        retriever_config.num_block_records, drop_remainder=True
    )
    blocks = tf.data.experimental.get_single_element(blocks_dataset)

  with tempfile.TemporaryDirectory() as tmp_dir:
    ############################################################################
    # Prepare the output file.
    ############################################################################
    tmp_dir = pathlib.Path(tmp_dir)
    h5_output_path = tmp_dir / "codes.h5"
    output_file = h5py.File(h5_output_path, "w")
    flags_dict = {
        flag.name: flag.value
        for flag in flags.FLAGS.flags_by_module_dict()[argv[0]]
    }
    utils.to_json_file(tmp_dir / "params.json", flags_dict)

    for split in keys:
      with utils.log_duration(
          LOGGER,
          "main",
          "Creating the output hdf5 file, embeddings."
      ):
        num_entries = len(eli5[split]["id"])
        if _FLAG_USE_SUBSET.value:
          num_entries = min(num_entries, _FLAG_SUBSET_AMOUNT.value)
        split_group = output_file.create_group(split)

      with utils.log_duration(
          LOGGER, "main", "Creating the output hdf5 file, retrieval."
      ):
        split_group.create_dataset(
            constants.CTH5Fields.distances,
            shape=(num_entries, _FLAG_NUM_RETRIEVALS.value),
            dtype=np.float32,
        )
        split_group.create_dataset(
            constants.CTH5Fields.gpt2_question_ids_inputs,
            shape=(num_entries, _FLAG_CONTEXT_SIZE.value),
            dtype=np.int32
        )
        if split != "test":
          split_group.create_dataset(
              constants.CTH5Fields.gpt2_answer_ids_inputs,
              shape=(num_entries, _FLAG_CONTEXT_SIZE.value),
              dtype=np.int32
          )

        split_group.create_dataset(
            constants.CTH5Fields.gpt2_retrieved_ids,
            shape=(
                num_entries,
                _FLAG_NUM_RETRIEVALS.value,
                _FLAG_MAX_LENGTH_RETRIEVALS.value,
            ),
            dtype=np.int32
        )

      with utils.log_duration(LOGGER, "main", "Loading the reference db."):
        checkpoint_path = os.path.join(
            retriever_config.query_embedder_path, "encoded", "encoded.ckpt"
        )

        reference_db_device = tf_utils.device_mapping().CPUs[0].name
        with tf.device(reference_db_device):
          reference_db = tf_utils.load_reference_db(
              checkpoint_path,
              variable_name="block_emb",
          )

    ############################################################################
    # Prep the encoder and the tokenizer
    ############################################################################
    with utils.log_duration(
        LOGGER, "main", "Loading the encoder model and the tokenizer."
    ):
      with strategy.scope():
        query_encoder = hub.load(retriever_config.query_embedder_path, tags={})
      encode_fn = _make_encode_fn(query_encoder)
      encode_fn_strategy_run = _make_encode_fn_strategy_run_fn(
          strategy=strategy,
          encode_fn=encode_fn,
          )

      vocab_file = os.path.join(
          retriever_config.query_embedder_path, "assets", "vocab.txt"
      )
      utils.check_exists(vocab_file)
      do_lower_case = query_encoder.signatures["tokenization_info"](
      )["do_lower_case"]
      tokenization_info = dict(
          vocab_file=vocab_file, do_lower_case=do_lower_case
      )

      tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(
          query_encoder, tokenization_info
      )

    ############################################################################
    # Preprocess the dataset
    ############################################################################

    cls_token_id = tf.cast(
        vocab_lookup_table.lookup(tf.constant("[CLS]")), tf.int32
    )
    sep_token_id = tf.cast(
        vocab_lookup_table.lookup(tf.constant("[SEP]")), tf.int32
    )
    transform = _make_transform_fn(
        bert_tokenizer=tokenizer,
        bert_cls_token_id=cls_token_id,
        bert_sep_token_id=sep_token_id,
    )

    with utils.log_duration(LOGGER, "main", "generating codes"):
      tqdm_splits = tqdm.tqdm(keys)
      for split in tqdm_splits:
        tqdm_splits.set_description(f"Split `{split}`")
        eli5: Dict[str, datasets.Dataset]
        write_start = 0

        if _FLAG_USE_SUBSET.value:
          _warn_subset(tqdm_splits)
          eli5[split] = eli5[split][:_FLAG_SUBSET_AMOUNT.value]
          utils.check_operator(
              operator.le, len(eli5[split]["id"]), _FLAG_SUBSET_AMOUNT.value
          )
          utils.check_operator(
              operator.le, len(eli5[split]["input"]), _FLAG_SUBSET_AMOUNT.value
          )
        else:
          utils.check_equal(len(eli5[split]), len(eli5[split]["id"]))
          utils.check_equal(len(eli5[split]), len(eli5[split]["input"]))

        if split != "test":
          for_slices = dict(
              sample_id=eli5[split]["id"],
              question=eli5[split]["input"],
              answer=[sample["answer"][0] for sample in eli5[split]["output"]]
          )
        else:
          for_slices = dict(
              sample_id=eli5[split]["id"],
              question=eli5[split]["input"],
          )

        ds = tf.data.Dataset.from_tensor_slices(for_slices)
        ds = ds.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
        ds = ds.map(_squeeze, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        tqdm_inner = tqdm.tqdm(
            enumerate(ds),
            total=len(eli5[split]["id"]) // _FLAG_BATCH_SIZE.value,
            desc=f"Split `{split}`: Batches"
        )

        for i, batch in tqdm_inner:
          ######################################################################
          # Enforce the current real batch size
          ######################################################################
          current_batch_size = batch["sample_id"].shape[0]
          for k, v in batch.items():
            utils.check_equal(v.shape[0], current_batch_size)
          ######################################################################

          gpt2_question_ids_inputs = _prep_field(
              batch["question"], gpt2_tokenizer
          )
          utils.check_equal(gpt2_question_ids_inputs.dtype, np.int32)
          utils.check_equal(
              gpt2_question_ids_inputs.shape[0], current_batch_size
          )

          if split != "test":
            gpt2_answer_ids_inputs = _prep_field(
                batch["answer"], gpt2_tokenizer
            )
            utils.check_equal(gpt2_answer_ids_inputs.dtype, np.int32)
            utils.check_equal(
                gpt2_answer_ids_inputs.shape[0], current_batch_size
            )

            assert len(gpt2_answer_ids_inputs.shape) == 2, (
                gpt2_answer_ids_inputs.shape
            )

          ######################################################################
          # Save the gpt2 tokenized question and answer
          ######################################################################
          end = write_start + current_batch_size

          utils.check_equal(
              output_file[split][
                  constants.CTH5Fields.gpt2_question_ids_inputs
              ][write_start:end].shape[0], current_batch_size)
          output_file[split][
              constants.CTH5Fields.gpt2_question_ids_inputs
          ][write_start:end] = gpt2_question_ids_inputs

          if split != "test":
            output_file[split][
                constants.CTH5Fields.gpt2_answer_ids_inputs
            ][write_start:end] = gpt2_answer_ids_inputs

          ######################################################################
          # Encode the samples.
          ######################################################################
          batch = strategy.experimental_distribute_values_from_function(
              tf_utils.make_dict_distribute_fn(batch)
          )

          embeddings = encode_fn_strategy_run(batch)
          embeddings = tf_utils.process_strat_output(
              embeddings, "embeddings", strategy, current_batch_size
          )
          utils.check_isinstance(embeddings, ops.EagerTensor)
          utils.check_equal(embeddings.shape[0], current_batch_size)

          # pytype doesn't seem to see that we check the type
          utils.check_equal(embeddings.shape[1], _FLAG_EMBEDDING_DEPTH.value)  # pytype: disable=attribute-error

          ######################################################################
          # Retrieve.
          ######################################################################
          with tf.device(reference_db_device):
            top_k, inner_prods = tf_utils.mips_exact_search(
                embeddings, _FLAG_NUM_RETRIEVALS.value, reference_db
            )
          top_k = tf_utils.process_strat_output(
              top_k, "top_k", strategy, current_batch_size
          )
          utils.check_equal(
              inner_prods.shape,
              (current_batch_size, _FLAG_NUM_RETRIEVALS.value)
          )
          utils.check_equal(
              top_k.shape, (current_batch_size, _FLAG_NUM_RETRIEVALS.value)
          )

          output_file[split]["distances"][write_start:end] = inner_prods

          gathered = tf.gather(blocks, top_k).numpy()
          utils.check_equal(gathered.shape[0], current_batch_size)

          utils.check_equal(write_start + gathered.shape[0], end)
          for j in range(gathered.shape[0]):
            local_gathered = gathered[j].tolist()
            utils.check_equal(len(local_gathered), _FLAG_NUM_RETRIEVALS.value)
            local_gathered = [sample.decode() for sample in local_gathered]
            token_ids = np.array(
                gpt2_tokenizer.batch_encode_plus(
                    local_gathered,
                    padding="max_length",
                    truncation=True,
                ).input_ids
            )
            for line in token_ids:
              assert not np.all(line == 0), line

            token_ids[token_ids == gpt2_tokenizer.eos_token_id] = -1
            output_file[split][constants.CTH5Fields.gpt2_retrieved_ids][
                write_start + j
            ] = token_ids[:, :_FLAG_MAX_LENGTH_RETRIEVALS.value]

          write_start += current_batch_size
    ############################################################################
    # Upload the results to GCS
    ############################################################################
    LOGGER.debug("DONE WITH THE PRODUCTION")
    output_file.close()
    with utils.log_duration(LOGGER, "main", "gsutil transfer"):
      command = [
          "/root/google-cloud-sdk/bin/gsutil", "-m", "cp", "-r",
          str(tmp_dir / "*"), target_path
      ]
      LOGGER.debug("Command: %s", " ".join(command))
      subprocess.check_call(command)
    LOGGER.debug("ALL DONE")


if __name__ == "__main__":
  app.run(main)
