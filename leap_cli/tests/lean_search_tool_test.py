# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Unit tests for lean_search_tool. All offline: HTTP is stubbed."""

import _path  # noqa: F401  (puts leap/ on sys.path)
import json
import os
import tempfile
import unittest
from unittest import mock

import lean_search_tool as ls


def _resp(payload):
  m = mock.MagicMock()
  m.read.return_value = json.dumps(payload).encode()
  m.__enter__.return_value = m
  return m


_LEANSEARCH_HIT = [[{"result": {
    "name": ["Real", "sqrt"],
    "module_name": ["Mathlib", "Data", "Real", "Basic"],
    "kind": "def", "type": "Real -> Real"}}]]
_LOOGLE_HIT = {"hits": [{"name": "irrational_sqrt_two",
                         "module": "Mathlib.NumberTheory.Real.Irrational",
                         "type": " Irrational\n  (√2)"}]}


class SearchTest(unittest.TestCase):

  def test_empty_query(self):
    res = ls.run_lean_search(ls.LeanSearchInput(query=""))
    self.assertFalse(res.success)
    self.assertIn("empty query", res.results_text.lower())

  @mock.patch("urllib.request.urlopen")
  def test_leansearch(self, urlopen):
    urlopen.return_value = _resp(_LEANSEARCH_HIT)
    res = ls.run_lean_search(ls.LeanSearchInput(query="sqrt"))
    self.assertTrue(res.success)
    self.assertIn("[DEF] Real.sqrt", res.results_text)
    self.assertIn("Mathlib.Data.Real.Basic", res.results_text)
    body = json.loads(urlopen.call_args.args[0].data)
    self.assertEqual(body, {"query": ["sqrt"], "num_results": 5})

  @mock.patch("urllib.request.urlopen")
  def test_auto_falls_back_to_loogle(self, urlopen):
    urlopen.side_effect = [OSError("HTTP 500"), _resp(_LOOGLE_HIT)]
    res = ls.run_lean_search(ls.LeanSearchInput(query="Irrational"))
    self.assertTrue(res.success)
    self.assertIn("irrational_sqrt_two", res.results_text)
    self.assertIn("results below are from loogle", res.results_text)
    self.assertIn("Irrational (√2)", res.results_text)  # whitespace collapsed

  @mock.patch("urllib.request.urlopen")
  def test_explicit_backend_does_not_fall_back(self, urlopen):
    urlopen.side_effect = OSError("HTTP 500")
    res = ls.run_lean_search(
        ls.LeanSearchInput(query="x", backend="leansearch"))
    self.assertFalse(res.success)
    self.assertEqual(urlopen.call_count, 1)

  @mock.patch("urllib.request.urlopen")
  def test_loogle_error_is_a_failure(self, urlopen):
    urlopen.return_value = _resp({"error": "Unknown identifier 'Foo'"})
    res = ls.run_lean_search(ls.LeanSearchInput(query="Foo", backend="loogle"))
    self.assertFalse(res.success)
    self.assertIn("Unknown identifier", res.error_message)


class QuotaTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.ok = ls.LeanSearchOutput(results_text="[DECL] X", success=True)

  def _main(self, quota, env):
    saved = {v: os.environ.pop(v) for v in ls._AGENT_ID_VARS if v in os.environ}
    self.addCleanup(os.environ.update, saved)
    with mock.patch.dict(os.environ, env), \
        mock.patch.object(ls, "run_lean_search", return_value=self.ok) as run:
      for _ in range(3):
        ls.main(["prog", "--query=X", f"--subagent_quota={quota}",
                 f"--quota_dir={self.tmp.name}"])
    return run.call_count

  def test_negative_is_unlimited(self):
    self.assertEqual(self._main(-1, {"LEAP_AGENT_ID": "a"}), 3)

  def test_zero_blocks(self):
    self.assertEqual(self._main(0, {"LEAP_AGENT_ID": "b"}), 0)

  def test_positive_is_enforced(self):
    self.assertEqual(self._main(2, {"LEAP_AGENT_ID": "c"}), 2)

  def test_antigravity_id_is_honoured(self):
    self.assertEqual(self._main(1, {"ANTIGRAVITY_CONVERSATION_ID": "d"}), 1)

  def test_no_agent_id_means_no_tracking(self):
    self.assertEqual(self._main(1, {}), 3)


if __name__ == "__main__":
  unittest.main()
