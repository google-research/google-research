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

"""Evaluate machine translation performance using BLEU and COMET."""
import click
from comet import download_model
from comet import load_from_checkpoint
from datasets import load_dataset
import evaluate
import transformers
from vllm import LLM
from vllm import SamplingParams


AutoTokenizer = transformers.AutoTokenizer


@click.command()
@click.option("-max_tokens", type=int, default=256)
@click.option("-visualize_text", type=bool, default=True)
@click.option(
    "-ckpt", type=str, default="gemma-7b-it-sft-dialogsum-longer/checkpoint-160"
)
def main(max_tokens, visualize_text, ckpt):

  # load in validation set
  vali_dataset = load_dataset(
      "json", data_files="data/mt_test.json", field="instances", split="train"
  )
  tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
  vali_data = [
      tokenizer.apply_chat_template(
          [{"role": "user", "content": ele}],
          tokenize=False,
          add_generation_prompt=True,
      )
      for ele in vali_dataset["instruction"]
  ]

  sampling_params = SamplingParams(
      max_tokens=max_tokens, temperature=0, top_p=1, stop=["\n"]
  )
  llm = LLM(model=ckpt)
  gen_outputs = llm.generate(vali_data, sampling_params)

  out_ls, ref_ls, data = [], [], []
  for inst, ele, ref in zip(vali_data, gen_outputs, vali_dataset["response"]):
    out_ls += [ele.outputs[0].text]
    ref_ls += [ref]
    data += [{
        "src": (
            inst.split(
                "Translate Assamese sentence into English. This is the Assamese"
                " sentence: "
            )[1].split(" Here is the English translation:")[0]
        ),
        "mt": ele.outputs[0].text,
        "ref": ref,
    }]
    if visualize_text:
      print("Output: ", ele.outputs[0].text)
      print("Reference: ", ref)
      print(">" * 50)

  bleu = evaluate.load("bleu")
  results = bleu.compute(predictions=out_ls, references=ref_ls)
  print("BLEU: ", results)

  model_path = download_model("Unbabel/wmt22-comet-da")
  model = load_from_checkpoint(model_path)
  model_output = model.predict(data, batch_size=8, gpus=1)
  print("COMET: ", model_output.system_score)


if __name__ == "__main__":
  main()
