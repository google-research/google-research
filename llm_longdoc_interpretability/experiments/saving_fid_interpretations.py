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

"""Top-level code for the FiD experiments."""

from collections.abc import Sequence
import json
import pickle
import time

from absl import app
from absl import flags
from datasets_and_prompts import get_prompt_list_from_doc
from datasets_and_prompts import return_shifts_from_prompt_list_v2
from interp_and_visualization import plot_big_dict
from t5_with_flash.modeling_t5_with_flash import fid_adjust_existing_model
from t5_with_flash.modeling_t5_with_flash import return_pretrained_model
import torch


_MAX_OUTPUT_LENGTH = flags.DEFINE_integer(
    "max_output_length", 20, "The maximum output length for LLM generation."
)

_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "FiD-t5-large",
    "The pretrained model's name. Can only be 'FiD-t5-large' for this script",
)

_QUERIES_WITH_RETRIEVALS_PATH = flags.DEFINE_string(
    "queries_with_retrievals_path",
    "contriever-main/contriever_msmarco_nq/nq-open-oracle-top1000.jsonl",
    (
        "The file path for the input jsonl containing all of the queries "
        " alongside their retrieved documents/ passages for context."
    ),
)

_RESULTS_PATH = flags.DEFINE_string(
    "results_path",
    "results_data/shap_shub_dicts_NQ/",
    (
        "A file path for the output pkl's containing all of the generated"
        " Shapley or Banzhaf value interpretations."
        "Either results_data/shap_shub_dicts_MIRACL/ or "
        "results_data/shap_shub_dicts_NQ/"
    ),
)

_MIN_SAMPLE_ID = flags.DEFINE_integer(
    "min_sample_id", 0, "The starting index for the samples to iterate over."
)
_MAX_SAMPLE_ID = flags.DEFINE_integer(
    "max_sample_id",
    1000,
    (
        "The ending index for the samples to iterate over."
        "At most 2655 for NQ and 2863 for MIRACL."
    ),
)

_NUMBER_OF_PASSAGES_TO_CHECK = flags.DEFINE_multi_integer(
    "number_of_passages_to_check",
    [1, 3, 5, 10, 20, 30, 40, 50],
    (
        "The number of passages to check for interpretability"
        " measurements.Default is to check up to 50 showing the value of the"
        " reranking at eachlevel of number of retrieved passages.  Also look at"
        " taking [200] or [500]to show the value and scalability to large"
        " numbers of passages."
    ),
)


_INTERPRETABILITY_VALUE_TYPE = flags.DEFINE_string(
    "interpretability_value_type",
    "shapley",
    (
        "The type of interpretability value to use.Default is to use Shapley."
        " Can be either 'shapley', 'banzhaf', or 'banzhaf10"
    ),
)

_INTERPRETABILITY_SAMPLES = flags.DEFINE_integer(
    "interpretability_samples",
    100,
    (
        "The number of samples to use in the approximation"
        "algorithms for interpretability."
        "For Shapley it is number of permutations."
        "For Banzhaf it is number of subsets."
        "Default is 100."
    ),
)
_DECODER_BATCH_SIZE = flags.DEFINE_integer(
    "decoder_batch_size",
    100,
    (
        "The number of output decodings to generate"
        "in parallel.  This is what gives the FiD"
        "version of this algorithm so much speed."
        "Optimal performance for FiD-large happens at"
        "100. Going to larger batches only improves"
        "performance marginally; increases memory"
        "consumption substantially."
    ),
)
_BANZHAF_PROBABILITY = flags.DEFINE_float(
    "banzhaf_probability",
    0.1,
    (
        "The probability to use for the Bernoulli"
        "distribution in the Banzhaf sampling."
        "Only used when mode is 'banzhaf10'."
    ),
)


# pylint: disable=invalid-name
def get_FiD_file_name(interpretability_mode, x):
  """Builds FiD file names."""
  file_name = "FiD_shap_shup_perm_dict_obj__"
  if interpretability_mode["mode"] == "shapley":
    file_name = (
        file_name
        + interpretability_mode["mode"]
        + "_P"
        + str(interpretability_mode["P"])
        + "_D"
        + str(interpretability_mode["D"])
        + "_x"
        + str(x)
        + ".pkl"
    )
  elif interpretability_mode["mode"] == "banzhaf":
    file_name = (
        file_name
        + interpretability_mode["mode"]
        + "_S"
        + str(interpretability_mode["S"])
        + "_D"
        + str(interpretability_mode["D"])
        + "_x"
        + str(x)
        + ".pkl"
    )
  elif interpretability_mode["mode"] == "banzhaf10":
    file_name = (
        file_name
        + interpretability_mode["mode"]
        + "_S"
        + str(interpretability_mode["S"])
        + "_p"
        + str(interpretability_mode["prob"])
        + "_D"
        + str(interpretability_mode["D"])
        + "_x"
        + str(x)
        + ".pkl"
    )
  return file_name


def create_fid_object_to_save(interpretability_mode, x, outputs, time_taken):
  """Builds dict of FiD experiment results."""
  parameters = {
      "x": x,
      "interpretability_mode": interpretability_mode,
  }
  big_dict, lil_dict, list_of_perms = outputs
  thing_to_save = {
      "big_dict": big_dict,
      "lil_dict": lil_dict,
      "list_of_perms": list_of_perms,
      "total_time_taken_each_z": [time_taken],
      "parameters": parameters,
  }
  return thing_to_save


def fid_prompt_preparation_no_clipping_at_end(
    prompt_style, ctx, question, model4, t, device, number_of_passages
):
  """Prepares the input_ids for the FiD prompt given the queries and context.

  Args:
    prompt_style: FiD_style prompt is required
    ctx: context documents
    question: query string
    model4: FiD T5 model to be preprepared (with sparsity matrix)
    t: tokenizer
    device: device of model4
    number_of_passages: maximum number of passages considered by FiD model

  Returns:
    the input ids to be passed to generation function
  """
  nop = number_of_passages
  _, prompt_list = get_prompt_list_from_doc(prompt_style, ctx, question)
  inp_ids = t(
      prompt_list,
      return_tensors="pt",
      padding=True,
      return_attention_mask=True,
  ).input_ids

  max_in_size = 256
  inp_ids = torch.nn.functional.pad(
      inp_ids, (0, max_in_size - inp_ids.shape[-1])
  )
  inp_ids = inp_ids[:, :max_in_size]
  inp_ids = inp_ids.to(device)
  inp_ids = inp_ids.reshape(1, -1)
  # assuming max_in = 256 and block = 128
  sparsity = torch.zeros((2 * nop, 2 * nop))
  for pp in range(nop):  # block diagonal sparsity
    sparsity[2 * pp : 2 * pp + 2, 2 * pp : 2 * pp + 2] = 1
  sparsity = sparsity.to(device)
  bias_shift_np, _, local_lengths_np = return_shifts_from_prompt_list_v2(
      prompt_list, 128
  )
  fid_adjust_existing_model(
      model4, device, sparsity, bias_shift_np, local_lengths_np, nop
  )
  return inp_ids


def construct_interpretability_mode_dictionary_from_arguments(
    mode, samples, decoder_batch_size, banzhaf_probability
):
  """Constructs dict containing interpretability information."""
  interpretability_mode = {}
  interpretability_mode["mode"] = mode
  assert mode in ["shapley", "banzhaf", "banzhaf10"], (
      "mode " + mode + " not supported"
  )
  if mode == "shapley":
    interpretability_mode["P"] = samples
  else:
    interpretability_mode["S"] = samples
  interpretability_mode["DBS"] = decoder_batch_size
  interpretability_mode["prob"] = banzhaf_probability
  return interpretability_mode


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  max_output_length = _MAX_OUTPUT_LENGTH.value
  model_name = _MODEL_NAME.value
  queries_with_retrievals_path = _QUERIES_WITH_RETRIEVALS_PATH.value
  results_path = _RESULTS_PATH.value

  min_x = _MIN_SAMPLE_ID.value
  max_x = _MAX_SAMPLE_ID.value
  xs = list(range(min_x, max_x))

  nop2s_to_check = _NUMBER_OF_PASSAGES_TO_CHECK.value
  max_num_of_passages = max(nop2s_to_check)

  interpretability_mode = (
      construct_interpretability_mode_dictionary_from_arguments(
          _INTERPRETABILITY_VALUE_TYPE.value,
          _INTERPRETABILITY_SAMPLES.value,
          _DECODER_BATCH_SIZE.value,
          _BANZHAF_PROBABILITY.value,
      )
  )

  all_times = []
  device = torch.device("cuda:0")
  model4, t = return_pretrained_model(model_name)
  start_load = time.time()
  model4 = model4.to(device)
  print("model loaded onto GPU in ", time.time() - start_load, "seconds")

  path = queries_with_retrievals_path
  with open(path, "r") as json_file:
    json_list = list(json_file)
  print("retrieved passages properly loaded")

  with torch.no_grad():
    # assuming max_in = 256 and block = 128
    sparsity = torch.zeros((2 * max_num_of_passages, 2 * max_num_of_passages))
    for pp in range(max_num_of_passages):
      sparsity[2 * pp : 2 * pp + 2, 2 * pp : 2 * pp + 2] = 1

    total_start_time = time.time()
    for x in xs:
      json_str = json_list[x]
      result = json.loads(json_str)
      ctx = result["ctxs"]
      question = result["question"]
      answers = result["answers"]

      prompt_style = "FiD_sparse"
      max_in_size = 256
      inp_ids = fid_prompt_preparation_no_clipping_at_end(
          prompt_style, ctx, question, model4, t, device, max_num_of_passages
      )

      for curr_num_of_passages in nop2s_to_check:
        interpretability_mode["D"] = curr_num_of_passages

        interp_start_time = time.time()
        print("generating interpretations")
        outputs = model4.jam_FiD_enc_interpretation(
            inp_ids[:, : curr_num_of_passages * max_in_size],
            max_length=max_output_length,
            output_scores=True,
            return_dict_in_generate=True,
            number_of_out_decodings=curr_num_of_passages,
            tokenizer=t,
            interpretability_mode=interpretability_mode,
        )
        time_taken = time.time() - interp_start_time
        print("interp_time_taken", time_taken)
        print()

        PLOTTING = False
        if PLOTTING:
          print("question", question)
          print("answers", answers)
          print()
          print()
          plot_big_dict(outputs[1], outputs[0])
          print()
          print()
          print()
          print()

        SAVING = True
        if SAVING:
          thing_to_save = create_fid_object_to_save(
              interpretability_mode, x, outputs, time_taken
          )
          file_name = get_FiD_file_name(interpretability_mode, x)
          pickle.dump(thing_to_save, open(results_path + file_name, "wb"))

      else:
        raise NotImplementedError("NOT IMPLEMENTED WITHOUT FID-STYLE RIGHT NOW")

    total_time_taken = time.time() - total_start_time
    all_times.append(total_time_taken)


if __name__ == "__main__":
  app.run(main)
