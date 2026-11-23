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

"""Tests for the `leap` command and its saved settings. Offline."""

import _path  # noqa: F401  (puts leap/ on sys.path)
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import cli
import config


class _Home(unittest.TestCase):
  """Points LEAP_HOME at a temp dir and isolates the verifier env vars."""

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    env = {k: v for k, v in os.environ.items() if k not in config.KNOWN_KEYS}
    env["LEAP_HOME"] = self.tmp.name
    patcher = mock.patch.dict(os.environ, env, clear=True)
    patcher.start()
    self.addCleanup(patcher.stop)


class ConfigTest(_Home):

  def test_save_load_and_remove(self):
    config.save({"LEAN_PROJECT_DIR": "/p", "LEAN_VERIFIER_API_KEY": "k"})
    self.assertEqual(config.load(), {"LEAN_PROJECT_DIR": "/p",
                                     "LEAN_VERIFIER_API_KEY": "k"})
    config.save({"LEAN_VERIFIER_API_KEY": ""})
    self.assertEqual(config.load(), {"LEAN_PROJECT_DIR": "/p"})
    self.assertEqual(os.stat(config.config_path()).st_mode & 0o777, 0o600)

  def test_unknown_key_is_refused(self):
    with self.assertRaises(KeyError):
      config.save({"PATH": "/evil"})

  def test_apply_never_overrides_environment(self):
    config.save({"LEAN_PROJECT_DIR": "/saved", "LEAN_VERIFIER_URL": "http://s"})
    os.environ["LEAN_PROJECT_DIR"] = "/from-env"
    config.apply()
    self.assertEqual(os.environ["LEAN_PROJECT_DIR"], "/from-env")
    self.assertEqual(os.environ["LEAN_VERIFIER_URL"], "http://s")

  def test_corrupt_file_reads_as_empty(self):
    os.makedirs(self.tmp.name, exist_ok=True)
    with open(config.config_path(), "w") as f:
      f.write("{not json")
    self.assertEqual(config.load(), {})


class LeanKiminaTest(_Home):

  def test_kimina_is_saved_and_probed(self):
    with mock.patch.object(cli, "_probe", return_value=True) as probe, \
        contextlib.redirect_stdout(io.StringIO()):
      rc = cli.main(["lean", "--kimina", "http://h:8000/", "--api-key", "abc"])
    self.assertEqual(rc, 0)
    probe.assert_called_once_with(expect_backend="kimina")
    with open(config.config_path()) as f:
      saved = json.load(f)
    self.assertEqual(saved, {"LEAN_VERIFIER_BACKEND": "kimina",
                             "LEAN_VERIFIER_URL": "http://h:8000",
                             "LEAN_VERIFIER_API_KEY": "abc"})

  def test_new_project_is_pinned_to_lean_version(self):
    with mock.patch.object(cli.shutil, "which", return_value="/bin/lake"), \
        mock.patch.object(cli, "_run") as run, \
        mock.patch.object(cli, "_probe", return_value=True), \
        mock.patch.object(cli, "_setup_any_folder", return_value=True), \
        contextlib.redirect_stdout(io.StringIO()):
      rc = cli.main(["lean"])
    self.assertEqual(rc, 0)
    project = os.path.join(self.tmp.name, "leap_lean")
    self.assertEqual(config.project_toolchain(project),
                     "leanprover/lean4:" + config.LEAN_VERSION)
    with open(os.path.join(project, "lakefile.toml")) as f:
      self.assertIn(f'rev = "{config.LEAN_VERSION}"', f.read())
    self.assertEqual([c.args[0][1:] for c in run.call_args_list],
                     [["update"], ["exe", "cache", "get"], ["build"]])
    self.assertEqual(config.load()["LEAN_PROJECT_DIR"], project)
    # Local Lean is the default, so no backend is saved; a URL set later is
    # tried first, with this project as the fallback.
    self.assertNotIn("LEAN_VERIFIER_BACKEND", config.load())

  def test_default_project_on_other_toolchain_is_refused(self):
    project = os.path.join(self.tmp.name, "leap_lean")
    os.makedirs(project)
    for name, text in (("lakefile.toml", ""),
                       ("lean-toolchain", "leanprover/lean4:v4.20.0\n")):
      with open(os.path.join(project, name), "w") as f:
        f.write(text)
    with mock.patch.object(cli.shutil, "which", return_value="/bin/lake"), \
        mock.patch.object(cli, "_run") as run, \
        contextlib.redirect_stdout(io.StringIO()):
      rc = cli.main(["lean"])
    self.assertEqual(rc, 1)
    run.assert_not_called()
    self.assertEqual(config.load(), {})

  def test_any_folder_writes_env_file_and_profile_once(self):
    home = os.path.join(self.tmp.name, "home")
    os.makedirs(home)
    bashrc = os.path.join(home, ".bashrc")
    with open(bashrc, "w") as f:
      f.write('export LEAN_PATH="/old"')  # no trailing newline
    project = os.path.join(self.tmp.name, "leap_lean")
    os.makedirs(project)
    with open(os.path.join(project, "lean-toolchain"), "w") as f:
      f.write(config.LEAN_TOOLCHAIN + "\n")
    with mock.patch.dict(os.environ, {"HOME": home}), \
        mock.patch.object(cli.os, "name", "posix"), \
        mock.patch.object(cli.shutil, "which", return_value="/bin/elan"), \
        mock.patch.object(cli, "_project_lean_path", return_value="/a:/b"), \
        mock.patch.object(cli, "_run") as run, \
        mock.patch.object(cli, "_probe_any_folder", return_value=True), \
        contextlib.redirect_stdout(io.StringIO()):
      self.assertTrue(cli._setup_any_folder("/bin/lake", project, True))
      self.assertTrue(cli._setup_any_folder("/bin/lake", project, True))
    run.assert_called_with(["/bin/elan", "default", config.LEAN_TOOLCHAIN])
    with open(os.path.join(self.tmp.name, "env")) as f:
      self.assertIn('export LEAN_PATH="/a:/b"', f.read())
    with open(bashrc) as f:
      lines = f.read().splitlines()
    self.assertEqual(lines[0], 'export LEAN_PATH="/old"')
    sources = [l for l in lines if l.endswith('/env"')]
    self.assertEqual(len(sources), 1)  # idempotent
    self.assertEqual(lines[-1], sources[0])  # after the old LEAN_PATH

  def test_any_folder_on_windows_sets_user_environment(self):
    winreg = mock.MagicMock()
    winreg.QueryValueEx.side_effect = OSError  # no previous LEAN_PATH
    key = winreg.OpenKey.return_value.__enter__.return_value
    ctypes = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"winreg": winreg, "ctypes": ctypes}), \
        contextlib.redirect_stdout(io.StringIO()):
      written = cli._persist_lean_path_windows("C:\\a;C:\\b")
    winreg.SetValueEx.assert_called_once_with(
        key, "LEAN_PATH", 0, winreg.REG_SZ, "C:\\a;C:\\b")
    ctypes.windll.user32.SendMessageTimeoutW.assert_called_once()
    self.assertEqual(written, [r"HKEY_CURRENT_USER\Environment\LEAN_PATH"])

  def test_elan_install_uses_powershell_on_windows(self):
    with mock.patch.object(cli.os, "name", "nt"), \
        mock.patch.object(cli.shutil, "which", return_value=None), \
        mock.patch.object(cli.subprocess, "run") as run, \
        contextlib.redirect_stdout(io.StringIO()):
      rc = cli.main(["lean", "-y"])
    self.assertEqual(rc, 1)  # `lake` still missing after the mocked install
    self.assertEqual(run.call_args.args[0], cli.ELAN_INSTALL_WINDOWS)
    self.assertNotIn("shell", run.call_args.kwargs)

  def test_existing_project_without_lakefile_is_refused(self):
    with mock.patch.object(cli.shutil, "which", return_value="/bin/lake"), \
        contextlib.redirect_stdout(io.StringIO()):
      rc = cli.main(["lean", "--project", self.tmp.name])
    self.assertEqual(rc, 1)
    self.assertEqual(config.load(), {})


class InitTest(_Home):

  def test_init_writes_agents_and_problems_dir(self):
    dest = os.path.join(self.tmp.name, "ws")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
      rc = cli.main(["init", "claude", "--dest", dest])
    self.assertEqual(rc, 0)
    self.assertTrue(os.path.exists(
        os.path.join(dest, ".claude", "agents", "orchestrator-lemma.md")))
    self.assertTrue(os.path.isdir(os.path.join(dest, "problems")))
    self.assertIn("leap lean", out.getvalue())  # no verifier configured yet
    self.assertIn("claude --agent orchestrator-lemma", out.getvalue())


class LoopTest(_Home):

  def test_codex_gets_positional_resume_and_skill_prefix(self):
    with mock.patch.object(cli.subprocess, "call", return_value=0) as call:
      cli.main(["loop", "codex", "exec", "$leap-prove Prove problems/a.lean"])
    cmd = call.call_args.args[0]
    env = call.call_args.kwargs["env"]
    self.assertTrue(cmd[1].endswith("safe_lemma_runner.sh"))
    self.assertEqual(cmd[2:], ["exec", "$leap-prove Prove problems/a.lean"])
    self.assertEqual(env["LEAP_AGENT_CLI"], "codex")
    self.assertEqual(env["LEAP_RESUME_FLAG"], "")
    self.assertEqual(env["LEAP_RESUME_PREFIX"], "$leap-prove ")


class ProbeTest(_Home):
  """`leap lean --kimina` must see the URL answer, not the local fallback."""

  def _probe(self, out, **kw):
    import lean_verifier_tool as lv  # pylint: disable=g-import-not-at-top
    with mock.patch.object(lv, "run_lean_verifier", return_value=out), \
        contextlib.redirect_stdout(io.StringIO()) as printed:
      ok = cli._probe(**kw)  # pylint: disable=protected-access
    return ok, printed.getvalue()

  def _out(self, **kw):
    import lean_verifier_tool as lv  # pylint: disable=g-import-not-at-top
    return lv.LeanVerifierOutput(is_valid=True, used_axioms=[], **kw)

  def test_kimina_answer_passes(self):
    ok, _ = self._probe(self._out(backend="kimina"), expect_backend="kimina")
    self.assertTrue(ok)

  def test_local_fallback_fails_a_kimina_setup(self):
    ok, printed = self._probe(
        self._out(backend="local", remote_error="connection refused"),
        expect_backend="kimina")
    self.assertFalse(ok)
    self.assertIn("connection refused", printed)

  def test_local_setup_accepts_local(self):
    ok, _ = self._probe(self._out(backend="local"))
    self.assertTrue(ok)


class DoctorTest(_Home):
  """`leap doctor` with local Lean as the default."""

  def _doctor(self, env):
    import leap_doctor  # pylint: disable=g-import-not-at-top
    os.environ.update(env)
    with mock.patch.object(leap_doctor.shutil, "which",
                           side_effect=lambda b: None if b == "lake" else "/b/" + b), \
        contextlib.redirect_stdout(io.StringIO()) as printed:
      rc = leap_doctor.main(["leap doctor", "--offline"])
    return rc, printed.getvalue()

  def test_nothing_set_fails_on_the_local_setup(self):
    rc, printed = self._doctor({})
    self.assertEqual(rc, 1)
    self.assertIn("[ok  ] verifier backend -- local Lean", printed)
    self.assertIn("[FAIL] lake", printed)
    self.assertIn("[FAIL] LEAN_PROJECT_DIR", printed)
    self.assertIn("leap lean", printed)

  def test_with_a_url_the_local_setup_is_only_a_fallback(self):
    rc, printed = self._doctor({"LEAN_VERIFIER_URL": "http://h:8000"})
    self.assertEqual(rc, 0)
    self.assertIn("kimina at http://h:8000", printed)
    self.assertIn("[warn] lake", printed)
    self.assertIn("[warn] LEAN_PROJECT_DIR", printed)
    self.assertNotIn("[FAIL]", printed)


if __name__ == "__main__":
  unittest.main()
