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

"""Dataset and model specific code.
"""
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from absl import flags
import constants
import dataclasses
import tensorflow as tf
import tf_utils
import transformers
import utils

# tf.config.run_functions_eagerly(True)


FLAGS = flags.FLAGS
LOGGER = logging.getLogger(__name__)

TokenizerType = Union[transformers.PreTrainedTokenizer,
                      transformers.PreTrainedTokenizerFast]


################################################################################
# Model Specific
################################################################################
@dataclasses.dataclass
class CreateModelReturn:
  tokenizer: TokenizerType
  model: Union[transformers.PreTrainedModel, List[transformers.PreTrainedModel]]
  strategy: Optional[tf.distribute.Strategy]


def load_model(
    model_load_path,
    model_key,
    distribute_mode,
    tpu_setup,
    num_replicas,
    ):
  """Tries to load the model.

  Logs duration and memory use. Logs additional information if loading the model
  fails.

  Args:
    model_load_path: Where to load the model from. Needs to be a **local** path.
    model_key: Key used to select the correct model loading function from
      the MODEL_FACTORIES dict.
    distribute_mode: A string describing how the model is distributed.
    tpu_setup: TPU configuration information.
    num_replicas: Number of data parallelism replicas.

  Returns:
    Returns an object containing the tokenizer, the model and the strategy.


  Raises:
    RuntimeError: If model_load_path points to nothing.
  """
  if distribute_mode not in constants.DistributeModeChoices.choices():
    raise ValueError(f"Unsupported distribute_mode: `{distribute_mode}`")

  if distribute_mode in constants.STRATEGIES:
    ##############################################################################
    # Model creation in case we are using tf.distribute.Strategies.
    ##############################################################################
    # We currently don't support GPU strategies, though adding them would be
    # simple.

    if distribute_mode == constants.DistributeModeChoices.tpustrategy:
      strategy = tf.distribute.TPUStrategy(
          tpu_setup.resolver,
      )
    elif distribute_mode == constants.DistributeModeChoices.onedevicestrategy:
      # Test mode with a single device, possibly a CPU.
      strategy = tf.distribute.OneDeviceStrategy(tf_utils.devices_to_use()[0])
    else:
      raise NotImplementedError(distribute_mode)

    with strategy.scope():
      config: CreateModelReturn = MODEL_FACTORIES[model_key](
          model_key,
          distribute_mode,
          None  # The replicas are created by the tf.distribute.Strategy obj
      )
      config.strategy = strategy

  else:
    ############################################################################
    # Model creation in the case we aren't using strategies.
    ############################################################################
    # In this case, most of the parallelism work is done inside of the specific
    # model creation functions.

    config: CreateModelReturn = MODEL_FACTORIES[model_key](
        model_load_path,
        model_key,
        distribute_mode,
        num_replicas,
    )
    config.strategy = None
  return config


def _create_gpt2(
    model_name,
    distribute_mode,
    num_replicas  # pylint: disable=unused-argument
):
  """Loads the tokenizer and the model for the GPT2 extra large model."""

  ##############################################################################
  # Load the tokenizer
  ##############################################################################
  LOGGER.debug("Loading the weights: `%s`", model_name)
  tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_name)
  LOGGER.debug("Done loading the tokenizer.")
  LOGGER.debug("Loading the model weights.")

  ##############################################################################
  # Build the model(s) if we are splitting the model between devices per replica
  ##############################################################################
  if distribute_mode in {
      constants.DistributeModeChoices.split_and_data_parallel,
      constants.DistributeModeChoices.split_vertically
  }:
    # TODO(julesgm): This part needs to be reworked.
    raise NotImplementedError()

    # target_devices_info = tf_utils.InformationOnDevices()
    ############################################################################
    # Build the model function
    ############################################################################
    # if tf_utils.devices_to_use()[0].device_type == "CPU":
    #
    #  # Edge case of testing on a CPU-only device. Mostly for local debugging.
    #  # Our compute node tooling doesn't work in this case.
    #   def make_model(data_parallelism_rank=0):
    # pylint: disable=unused-argument
    #     model = modeling_tf_gpt2_model_par.
    #     TFGPT2LMHeadModel.from_pretrained(
    #         tf_model_path,
    #         config=config,
    #         cache_dir=cache_dir,
    #         devices=tf_utils.devices_to_use(),
    #         cpu=tf_utils.devices_to_use()[0],
    #     )
    #     return model
    # else:
    #   # Regular case with GPUs or TPUs.
    #   def make_model(data_parallelism_rank=0):
    #     # TODO(julesgm): This part needs work.
    #     model = modeling_tf_gpt2_model_par.
    #     TFGPT2LMHeadModel.from_pretrained(
    #         tf_model_path,
    #         config=config,
    #         cache_dir=cache_dir,
    #         devices=target_devices_info.devices_by_device_id[
    #             data_parallelism_rank],
    #         cpu=tf_utils.device_mapping().CPUs[1],
    #     )
    #     return model

    # ############################################################################
    # # Build the model(s)
    # ############################################################################
    # if distribute_mode == constants.
    # DistributeModeChoices.split_and_data_parallel:
    #   # Multiple instances if we are doing data parallelism
    #   if num_replicas > target_devices_info.num_devices:
    #     raise ValueError("num_replicas larger than "
    #                      "target_devices_info.num_devices. \n"
    #                      f" - num_replicas: {num_replicas} \n"
    #                      f" - num_devices:
    #                      {target_devices_info.num_devices}")
    #   model = [make_model(rank) for rank
    #            in range(num_replicas)]
    # else:
    #   model = make_model()

  ##############################################################################
  # Build the model instance otherwise
  ##############################################################################
  else:
    with utils.log_duration(LOGGER, "main", "Loading the model."):
      model = transformers.TFGPT2LMHeadModel.from_pretrained(
          model_name,
          )

  logging.debug("Done loading the %s model.", model_name)
  return CreateModelReturn(
      tokenizer=tokenizer,
      model=model,
      strategy=None,
  )


################################################################################
# Dataset Specific
################################################################################
def create_lm_ds_kilt_eli5(
    *,
    tokenizer,
    context_window_size,  # pylint: disable=unused-argument
    dataset_name,  # pylint: disable=unused-argument
    batch_size,
    split,
    db_path,  # pylint: disable=unused-argument
    random_seed,
    use_subset,  # pylint: disable=unused-argument
    subset_size,  # pylint: disable=unused-argument
    repeat,
    use_helper_words,
    approach_type,
    retriever,
    num_retrievals,
    retrieval_temperature,
    enable_debug_checks,
    retrieval_bank_size,  # pylint: disable=unused-argument
    dataset_type,
    qty_shuffle,
    tfr_prefix,
    max_length_generation,
):
  """Dataset preparation function for the Kilt version of the ELI5 dataset.

  This is for when the dataset is consumed by language models.

  Args:
    tokenizer: Tokenizer of the reader model.
    context_window_size: Size of the context of the reader model.
      Not used here.
    dataset_name: Exact name of the dataset. Some datasets share the same
      function, with small specific differences. Not used here.
    batch_size: Size of the batch for the reader model.
    prefetch_size: How many batches to prefetch.
    split: The train, evaluation or test split.
    dataset_paths_root: Root directory of the datasets. Not used here.
    random_seed: Seed used to shuffle the dataset. Should change at each epoch.
    use_subset: Whether to use a subset of the data
    subset_size: Size of the subset
    repeat: Whether to repeat the dataset
    use_helper_words: Whether to add helper words in the merged samples.
    approach_type: Type of overall solution we are using.
    retriever: Object that does the retrieval.
    num_retrievals: Number of retrievals to do.
    retrieval_temperature: For the retrieval methods that do sampling, what
      temperature to use.
  Returns:
    A tf.data.Dataset object that generates input_ids and label_ids for the
    generator model.
  Raises:
    RuntimeError: If we didn't find any files with the glob pattern.
    RuntimeError: If we are using a dataset type that is not supported.
  """

  maybe_retrieve_and_merge = _make_maybe_retrieve_and_merge_fn(
      tokenizer=tokenizer,
      context_size=context_window_size,
      retriever=retriever,
      temperature=retrieval_temperature,
      num_retrievals=num_retrievals,
      ds_split=split,
      approach_type=approach_type,  # FLAG_APPROACH_TYPE.value
      use_helper_words=use_helper_words,  # FLAG_USE_HELPER_WORDS
      enable_debug_checks=enable_debug_checks,
      max_length_generation=max_length_generation,
  )
  if dataset_type == constants.DatasetTypeChoices.hdf5:
    raise ValueError("The hdf5 dataset type is not supported anymore."
                     "It is strictly worse than tfr.")
    #
    # with utils.log_duration(LOGGER, "create_lm_ds_kilt_eli5",
    # "loading codes.h5"):
    #   input_file = h5py.File(tf.io.gfile.GFile(db_path, "rb"),
    #   "r")[split]
    #
    # if use_subset:
    #   new = {}
    #   for k, v in input_file.items():
    #     new[k] = v[:subset_size]
    #   input_file = new
    #
    # def load(field_name):
    #   if field_name == constants.CTH5Fields.gpt2_retrieved_ids:
    #     return input_file[field_name][:, :retrieval_bank_size]
    #   else:
    #     return input_file[field_name]
    #
    # with utils.log_duration(
    #     LOGGER, "create_lm_ds_kilt_eli5", "gpt2_question_ids_inputs"
    # ):
    #   gpt2_question_ids_inputs = load(
    #       constants.CTH5Fields.gpt2_question_ids_inputs
    #   )
    #
    # with utils.log_duration(
    #     LOGGER,
    #     "create_lm_ds_kilt_eli5",
    #     constants.CTH5Fields.gpt2_answer_ids_inputs
    # ):
    #   answer_ids_inputs = load(
    #       constants.CTH5Fields.gpt2_answer_ids_inputs
    #   )
    #
    # stacks = {
    #     constants.CTH5Fields.gpt2_question_ids_inputs:
    #     gpt2_question_ids_inputs,
    #     constants.CTH5Fields.gpt2_answer_ids_inputs:
    #     answer_ids_inputs,
    # }
    #
    # if approach_type == constants.ApproachTypeChoices.cached_pretok:
    #   with utils.log_duration(
    #       LOGGER, "create_lm_ds_kilt_eli5", constants.CTH5Fields.distances
    #   ):
    #     stacks[constants.CTH5Fields.distances] = load(
    #         constants.CTH5Fields.distances
    #     )
    #   with utils.log_duration(
    #       LOGGER,
    #       "create_lm_ds_kilt_eli5",
    #       constants.CTH5Fields.gpt2_retrieved_ids
    #   ):
    #     stacks[constants.CTH5Fields.gpt2_retrieved_ids] = load(
    #         constants.CTH5Fields.gpt2_retrieved_ids,
    #         retrieval_bank_size=retrieval_bank_size,
    #     )
    #
    # LOGGER.debug("from_tensor_slices")
    #
    # ds = tf.data.Dataset.from_tensor_slices(stacks)
  elif dataset_type == constants.DatasetTypeChoices.tfr:
    glob_pattern = os.path.join(tfr_prefix, f"{split}*")
    filenames = list(tf.io.gfile.glob(glob_pattern))
    if not filenames:
      raise RuntimeError(
          f"filnames is empty. Glob pattern was: {glob_pattern}"
      )

    ds = tf.data.TFRecordDataset(
        filenames=filenames,
        num_parallel_reads=tf.data.experimental.AUTOTUNE,
    )

    description: Dict[str, tf.io.FixedLenFeature] = {
        constants.CTH5Fields.distances:
            tf.io.FixedLenFeature((), tf.string),
        constants.CTH5Fields.gpt2_retrieved_ids:
            tf.io.FixedLenFeature((), tf.string),
        constants.CTH5Fields.gpt2_question_ids_inputs:
            tf.io.FixedLenFeature((), tf.string),
    }
    if split != constants.SplitChoices.test:
      description[
          constants.CTH5Fields.gpt2_answer_ids_inputs
      ] = tf.io.FixedLenFeature((), tf.string)

    feature_dtypes: Dict[str, tf.dtypes] = {
        constants.CTH5Fields.distances:
            tf.float32,
        constants.CTH5Fields.gpt2_retrieved_ids:
            tf.int32,
        constants.CTH5Fields.gpt2_question_ids_inputs:
            tf.int32,
    }
    if split != constants.SplitChoices.test:
      feature_dtypes[
          constants.CTH5Fields.gpt2_answer_ids_inputs
      ] = tf.int32

    feature_shape: Dict[str, Tuple[int, Ellipsis]] = {
        constants.CTH5Fields.distances:
            (10,),
        constants.CTH5Fields.gpt2_retrieved_ids:
            (10, context_window_size,),
        constants.CTH5Fields.gpt2_question_ids_inputs:
            (context_window_size,),
    }
    if split != constants.SplitChoices.test:
      feature_shape[constants.CTH5Fields.gpt2_answer_ids_inputs] = (
          context_window_size,
      )

    @tf.function
    def parse(sample):
      example = tf.io.parse_single_example(sample, description)
      output = {}
      for k, v in example.items():
        output[k] = tf.io.parse_tensor(v, out_type=feature_dtypes[k])
        output[k].set_shape(feature_shape[k])
      return output

    ds = ds.map(
        parse,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False
        )
  else:
    raise RuntimeError(dataset_type)

  if repeat:
    ds = ds.repeat()

  utils.check_not_none(random_seed)
  utils.check_not_none(qty_shuffle)
  ds = ds.shuffle(qty_shuffle, seed=random_seed)

  ds = ds.batch(
      batch_size,
      drop_remainder=split != constants.SplitChoices.test
  )

  # We can't use parallel calls here, the huggingface Rust fast tokenizer
  # breaks with multiple threads. It seems to still be worth it over their
  # slow one though, vs using parallel threads.
  ds = ds.map(maybe_retrieve_and_merge,)

  return ds.prefetch(tf.data.experimental.AUTOTUNE)


def _make_maybe_retrieve_and_merge_fn(
    *,
    tokenizer,
    context_size,
    ds_split,
    approach_type,  # FLAG_APPROACH_TYPE.value
    use_helper_words,  # FLAG_USE_HELPER_WORDS
    retriever,  # pylint: disable=unused-argument
    temperature,
    num_retrievals,
    enable_debug_checks,
    max_length_generation,
    tf_function_kwargs = None,
):
  """Build the `maybe_retrieve_and_merge` closure."""
  tf_function_kwargs = {} if tf_function_kwargs is None else tf_function_kwargs
  not_test_split = ds_split != constants.SplitChoices.test

  @tf.function(**tf_function_kwargs)
  def maybe_retrieve_and_merge(
      batch,
  ):
    """Retrieve if needed, then finalize the prep. for model consumption."""

    batch_size = tf.shape(batch[
        constants.CTH5Fields.gpt2_question_ids_inputs
    ])[0]

    # Prepare the question ids inputs
    question_ids_inputs = batch[constants.CTH5Fields.gpt2_question_ids_inputs]
    question_ids_inputs = tf.RaggedTensor.from_tensor(
        question_ids_inputs,
        padding=constants.RAGGED_PADDING_ID
    )

    # Prepare the answer ids inputs
    answer_ids_inputs = None
    answer_ids_labels = None
    if not_test_split:
      answer_ids_inputs = batch[constants.CTH5Fields.gpt2_answer_ids_inputs]
      answer_ids_inputs = tf.RaggedTensor.from_tensor(
          answer_ids_inputs,
          padding=constants.RAGGED_PADDING_ID
      )
      answer_ids_labels = answer_ids_inputs

    ############################################################################
    # Prepare the helper words
    ############################################################################
    helper_word_token_ids = None
    if use_helper_words:

      helper_text = {"question": "Question:\n",
                     "context": "\nContext:\n",
                     "answer": "\nAnswer:\n"
                     }
      helper_word_token_ids = {}
      for k in helper_text:
        ids = tf.constant(tokenizer.encode(helper_text[k]), dtype=tf.int32)
        ids = tf.repeat(tf.expand_dims(ids, 0), batch_size, axis=0)
        helper_word_token_ids[k] = ids
      question_ids_inputs = tf.concat(
          [helper_word_token_ids["question"], question_ids_inputs],
          axis=1
      )

    ##########################################################################
    # W/ Cached Retrievals
    ##########################################################################
    label_ids = None
    if approach_type == constants.ApproachTypeChoices.cached_pretok:
      bpe_indices_gpt2 = batch[constants.CTH5Fields.gpt2_retrieved_ids]
      bpe_indices_gpt2 = tf.RaggedTensor.from_tensor(
          bpe_indices_gpt2,
          ragged_rank=2,
          padding=constants.RAGGED_PADDING_ID
      )

      distances = batch[constants.CTH5Fields.distances]
      input_ids, label_ids = _prepare_samples_w_retrieval(
          split=ds_split,
          batch_size=batch_size,
          question_ids_inputs=question_ids_inputs,
          answer_ids_inputs=(
              answer_ids_inputs if not_test_split else None
          ),
          gpt2_tokenized_retrieved=bpe_indices_gpt2,
          num_retrievals=num_retrievals,
          temperature=temperature,
          context_size=context_size,
          enable_debug_checks=enable_debug_checks,
          distances=distances,
          max_generation_length=max_length_generation,
          helper_word_token_ids=(
              helper_word_token_ids if use_helper_words else None
          ),
          use_helper_words=use_helper_words,
      )

    elif approach_type == constants.ApproachTypeChoices.naked_lm:
      ##########################################################################
      # Without Retrievals
      ##########################################################################
      if use_helper_words:
        question_ids_inputs = tf.concat([
            question_ids_inputs,
            helper_word_token_ids["answer"],
        ], axis=1)

      question_ids_labels = tf.ones_like(
          question_ids_inputs
      ) * constants.PPL_MASK_ID

      if not_test_split:
        input_ids = tf.concat((question_ids_inputs, answer_ids_inputs),
                              axis=1)
        label_ids = tf.concat((question_ids_labels, answer_ids_labels),
                              axis=1)
      else:
        input_ids = question_ids_inputs
    else:
      raise RuntimeError("Unnsupported approach_type value"
                         f" {approach_type}")

    ############################################################################
    # Finalize the preparation
    ############################################################################
    # Convert to dense tensors
    input_ids = input_ids.to_tensor(tokenizer.eos_token_id)

    if not_test_split:
      final_eos = tf.RaggedTensor.from_tensor(
          tokenizer.eos_token_id * tf.ones([batch_size, 1], dtype=tf.int32)
      )
      label_ids = tf.concat([label_ids, final_eos], axis=1)
      label_ids = label_ids.to_tensor(constants.PPL_MASK_ID)

    # All samples need to have at least one token != -100 (PPL_MASK_ID)
    if enable_debug_checks and not_test_split:
      not_any_padding = tf.reduce_any(
          label_ids != constants.PPL_MASK_ID, axis=1
      )
      none_has_padding = tf.math.reduce_all(
          not_any_padding
      )
      qty_doesnt_have_padding = tf.reduce_sum(
          tf.cast(not_any_padding))

      check_no_padding = tf.Assert(
          none_has_padding,
          [qty_doesnt_have_padding]
      )
      with tf.control_dependencies([check_no_padding]):
        label_ids = tf.identity(label_ids)

    # Limit size
    input_ids = input_ids[:, :context_size]
    if not_test_split:
      label_ids = label_ids[:, :context_size]

    ############################################################################
    # Pad `input_ids` and `label_ids` to context_size
    ############################################################################
    # Prepare the ones
    pad_qty = tf.math.maximum(
        0, tf.constant(context_size) - tf.shape(input_ids)[1]
    )
    padding_ones = tf.ones(
        [batch_size, pad_qty],
        dtype=input_ids.dtype
    )
    # Pad the inputs
    input_padding = tokenizer.eos_token_id * padding_ones
    input_ids = tf.concat((input_ids, input_padding), axis=1)

    # Pad the labels labels
    if not_test_split:
      pad_qty = tf.math.maximum(
          0, tf.constant(context_size) - tf.shape(label_ids)[1]
      )
      padding_ones = tf.ones(
          [batch_size, pad_qty],
          dtype=input_ids.dtype
      )
      label_padding = -100 * padding_ones
      label_ids = tf.concat((label_ids, label_padding), axis=1)

    # Make checks
    if enable_debug_checks:
      control_dependencies = []
      control_dependencies.append(tf.Assert(
          tf.math.reduce_all(input_ids != -1),
          [input_ids],
          name="NoMinusOnesInputs"
      ))
      if not_test_split:
        control_dependencies.append(tf.Assert(
            tf.math.reduce_all(label_ids != -1),
            [label_ids],
            name="NoMinusOnesLabel"
        ))
        control_dependencies.append(tf.Assert(
            tf.logical_not(
                tf.math.reduce_any(
                    tf.math.reduce_all(label_ids != -100, axis=1)
                )
            ),
            [label_ids],
            name="NotAllMinusOneHundred"
        ))
      with tf.control_dependencies(control_dependencies):
        input_ids = tf.identity(input_ids)

    return dict(
        input_ids=input_ids,
        label_ids=label_ids if not_test_split else None
    )

  return maybe_retrieve_and_merge


@tf.function
def _tokenize_and_concat_while_loop(
    all_retrieved_tokens,
    indices,
    num_retrieved,
    batch_size,
):
  """Tokenizes and puts together the retrievals, per batch unit."""
  def condition(
      index,
      _  # pylint: disable=unused-argument
  ):
    return tf.less(index, num_retrieved)

  def body(
      index,
      concat_tokens,
  ):

    addition = tf.gather(all_retrieved_tokens, indices[:, index], batch_dims=1)

    concat_tokens = tf.concat([
        concat_tokens, addition
    ], axis=1)

    return index + 1, concat_tokens

  if batch_size is None:
    raise RuntimeError("batch_size is `None`. This should not happen.")

  return tf.while_loop(
      condition, body, [
          0, tf.RaggedTensor.from_tensor(
              tf.zeros(
                  shape=(batch_size, 0),
                  dtype=tf.int32
              ),
          )
      ])[1]


@tf.function
def _prepare_samples_w_retrieval(
    split,
    batch_size,
    question_ids_inputs,
    answer_ids_inputs,
    gpt2_tokenized_retrieved,
    distances,
    num_retrievals,
    temperature,
    context_size,
    enable_debug_checks,
    use_helper_words,
    helper_word_token_ids,
    max_generation_length
):
  """Prepares the samples that use retrieval."""
  assert (split == constants.SplitChoices.test) == (
      answer_ids_inputs is None
  ), (split == constants.SplitChoices.test, answer_ids_inputs)
  # If and only if

  is_not_test = split != constants.SplitChoices.test

  if not isinstance(question_ids_inputs, tf.RaggedTensor):
    question_ids_inputs = tf.RaggedTensor.from_tensor(
        question_ids_inputs,
        padding=constants.RAGGED_PADDING_ID
    )

  if enable_debug_checks:
    asserts = []
    asserts.append(
        tf.Assert(
            tf.math.reduce_all(
                question_ids_inputs != constants.RAGGED_PADDING_ID,
            ),
            [question_ids_inputs.to_tensor()]
        )
    )
    if is_not_test:
      asserts.append(
          tf.Assert(
              tf.math.reduce_all(
                  answer_ids_inputs != constants.RAGGED_PADDING_ID,
              ),
              [answer_ids_inputs.to_tensor()]
          )
      )
    with tf.control_dependencies(asserts):
      question_ids_inputs = tf.identity(question_ids_inputs)

  # These checks are at graph composition time, so OK
  utils.check_isinstance(question_ids_inputs, tf.RaggedTensor)

  if is_not_test:
    utils.check_isinstance(answer_ids_inputs, tf.RaggedTensor)

  ##############################################################################
  # Sample from the possible retrievals
  ##############################################################################
  # Choose the indices
  indices = tf_utils.sample_without_replacement(
      distances / temperature, num_retrievals
  )

  # Concatenate the retrievals
  concat_retrieved = _tokenize_and_concat_while_loop(
      gpt2_tokenized_retrieved,
      indices=indices,
      batch_size=batch_size,
      num_retrieved=num_retrievals,
  )

  # Add Context and Answer Helper Words
  if use_helper_words:
    concat_retrieved = tf.concat([
        helper_word_token_ids["context"],
        concat_retrieved,
    ], axis=1)

  # Cut the lengths down to max_lens_retrieval.
  # The eventual length of the ["question"] helper_tokens is included in
  # question_ids_inputs.
  if is_not_test:
    max_lens_retrieval = (
        context_size * tf.ones(
            shape=(batch_size,),
            dtype=tf.int64,
        )
        - (question_ids_inputs.row_lengths() +
           # We always generate the same length of text.
           max_generation_length +  # answer_ids_inputs.row_lengths() +
           (helper_word_token_ids["answer"].shape[1] if use_helper_words else 0)
           )
    )

  else:
    max_lens_retrieval = (
        context_size * tf.ones(
            shape=(batch_size,),
            dtype=tf.int64,
        ) - (question_ids_inputs.row_lengths()  +
             max_generation_length +
             (helper_word_token_ids["answer"].shape[1]
              if use_helper_words else 0
              )
             )
    )

  concat_retrieved = tf.ragged.boolean_mask(
      concat_retrieved,
      (
          tf.ragged.range(concat_retrieved.row_lengths()) <
          tf.expand_dims(max_lens_retrieval, axis=1)
      )
  )

  if enable_debug_checks:
    asserts = [
        tf.Assert(
            tf.math.reduce_all(max_lens_retrieval < context_size),
            [max_lens_retrieval, context_size]
        ),
    ]
    with tf.control_dependencies(asserts):
      concat_retrieved = tf.identity(concat_retrieved)

  if use_helper_words:
    if is_not_test:
      new_input_ids = tf.concat(
          [question_ids_inputs,
           concat_retrieved,
           helper_word_token_ids["answer"],
           answer_ids_inputs
           ],
          axis=1
      )
      new_label_ids = tf.concat(
          [-100 * tf.ones_like(question_ids_inputs),
           -100 * tf.ones_like(concat_retrieved),
           -100 * tf.ones_like(helper_word_token_ids["answer"]),
           answer_ids_inputs
           ],
          axis=1
      )
    else:
      new_input_ids = tf.concat(
          [question_ids_inputs,
           concat_retrieved,
           helper_word_token_ids["answer"],
           ],
          axis=1
      )
  else:
    if is_not_test:
      new_input_ids = tf.concat(
          [question_ids_inputs,
           concat_retrieved,
           answer_ids_inputs
           ],
          axis=1
      )
      new_label_ids = tf.concat(
          [-100 * tf.ones_like(question_ids_inputs),
           -100 * tf.ones_like(concat_retrieved),
           answer_ids_inputs
           ],
          axis=1
      )
    else:
      new_input_ids = tf.concat(
          [question_ids_inputs,
           concat_retrieved,
           ],
          axis=1
      )
  return new_input_ids, new_label_ids if is_not_test else None


################################################################################
# Varia
################################################################################

DATASET_CARDINALITIES = {
    constants.DatasetNameChoices.kilt_eli5: {
        "train": 272637,
        "eval": 1507,
        "test": 600,
    }
}

# Pick the correct model creation function from the Hugging Face Model key.
MODEL_FACTORIES = {
    "gpt2": _create_gpt2,
    "gpt2-medium": _create_gpt2,
    "gpt2-large": _create_gpt2,
    "gpt2-xl": _create_gpt2,
    "distilgpt2": _create_gpt2,
}


