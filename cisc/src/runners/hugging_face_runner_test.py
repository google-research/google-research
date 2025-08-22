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

"""Test for hugging_face_runner which are based on a fake tokenizer and model."""

from absl.testing import absltest
import torch
import transformers
from cisc.src.runners import hugging_face_runner


class FakeTokenizer:
  """Fake tokenizer the return the `ord` of letters with additional padding.

  That is, tokenizer("a b c") -> [padding..., ord('a'), ord('b'), ord('c')]
  """

  pad_size = 10
  padding_side = "left"

  def __call__(
      self,
      texts,
      add_special_tokens,
      padding=False,
      return_tensors="not-pt",
      *args,
      **kwargs,
  ):
    assert not add_special_tokens
    input_ids = []
    for text in texts:
      lst = [ord(c) for c in text.split(" ")]
      if padding:
        lst = [0] * (self.pad_size - len(lst)) + lst
      input_ids.append(lst)
    output = transformers.BatchEncoding()
    if return_tensors == "pt":
      input_ids = torch.tensor(input_ids)
    output.input_ids = input_ids
    output.attention_mask = None
    return output


# The logits to return for each token from the fake model.
_FAKE_LOGIT = torch.arange(200, dtype=float)
_FAKE_LOGIT_LOG_SOFTMAX = torch.nn.functional.log_softmax(_FAKE_LOGIT, dim=0)


class FakeModel:
  """Fake model that just returns the `_FAKE_LOGIT` for each token."""

  def __call__(self, input_ids, *args, **kwargs):
    output = torch.zeros(
        input_ids.shape[0], input_ids.shape[1], len(_FAKE_LOGIT)
    )
    output[:, :, :] = _FAKE_LOGIT
    return transformers.modeling_outputs.CausalLMOutput(logits=output)


class HuggingFaceRunnerTest(absltest.TestCase):

  def test_multi_token_log_prob(self):
    x = hugging_face_runner.get_completion_likelihood_multi_token(
        "a ", ["b c", "c"], FakeModel(), FakeTokenizer(), device="cpu"
    )

    self.assertAlmostEqual(
        x[0][0].item(), _FAKE_LOGIT_LOG_SOFTMAX[ord("b")].item(), places=3
    )
    self.assertAlmostEqual(
        x[0][1].item(), _FAKE_LOGIT_LOG_SOFTMAX[ord("c")].item(), places=3
    )
    self.assertAlmostEqual(
        x[1].item(), _FAKE_LOGIT_LOG_SOFTMAX[ord("c")].item(), places=3
    )


# # Sainty checks for a real model
# x = get_completion_likelihood(
#     "I", ["play piano", "bla soccer", "play wood bla"],
#     runner.model, runner.tokenizer
# )
# assert len(x[0]) == 2
# assert len(x[1]) == 2
# assert len(x[2]) == 3
# assert x[0][0] == x[2][0]  # Both represent the word "play"
# assert x[0][0] > x[1][0]  # "I play" is more likely than "I bla"
# assert x[0][1] > x[2][1]  # "I play piano" is more likely than "I play bla"

if __name__ == "__main__":
  absltest.main()
