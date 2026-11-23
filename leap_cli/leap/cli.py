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

"""The `leap` command and the console entry points the agents call.

  leap lean                 one-time: install Lean (elan) + a Mathlib project,
                            pinned to config.LEAN_VERSION, usable from any folder
  leap lean --kimina URL    ...or a kimina-lean-server, used while it answers
                            (local Lean otherwise)
  leap init claude          put the agents + skills into the current directory
  leap doctor               check everything is wired up
  leap loop claude ...      keep an agent CLI running until the proof is done
  leap config               show saved settings

The tools-only `leap-tools` wheel ships this module without the agents, skills
and runner, so it offers only `lean`, `doctor` and `config`.

The modules in this package import each other flatly (`import dag`), as they
did when run as scripts, so every entry point puts the package directory on
sys.path and runs the module as `__main__`.
"""

import argparse
import os
import runpy
import tempfile
import shutil
import subprocess
import sys
from typing import List, Optional

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if PKG_DIR not in sys.path:
  sys.path.insert(0, PKG_DIR)

import config  # pylint: disable=g-import-not-at-top,wrong-import-position

ELAN_INSTALL = ("curl -sSf https://elan.lean-lang.org/elan-init.sh"
                " | sh -s -- -y --default-toolchain none")
# Windows has no `sh`; elan ships a PowerShell installer with the same options.
ELAN_INSTALL_WINDOWS = [
    "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
    "$f = Join-Path $env:TEMP 'elan-init.ps1';"
    " Invoke-WebRequest https://elan.lean-lang.org/elan-init.ps1 -OutFile $f;"
    " & $f -NoPrompt 1 -DefaultToolchain none; Remove-Item $f"]
HOSTS = ("claude", "codex", "antigravity")
HAS_AGENTS = os.path.isdir(os.path.join(PKG_DIR, "agents"))
HAS_RUNNER = os.path.exists(os.path.join(PKG_DIR, "safe_lemma_runner.sh"))


# --------------------------------------------------------------------------
# Entry points for the three tools the agents run.
# --------------------------------------------------------------------------

def _run_module(module: str, prog: str) -> None:
  config.apply()
  sys.argv[0] = prog
  runpy.run_path(os.path.join(PKG_DIR, module + ".py"), run_name="__main__")


def state_manager_bin() -> None:
  _run_module("state_manager", "state_manager_bin")


def lean_verifier_tool_bin() -> None:
  _run_module("lean_verifier_tool", "lean_verifier_tool_bin")


def lean_search_tool_bin() -> None:
  _run_module("lean_search_tool", "lean_search_tool_bin")


# --------------------------------------------------------------------------
# leap lean
# --------------------------------------------------------------------------

def _say(msg: str) -> None:
  print(f"==> {msg}", flush=True)


def _run(cmd: List[str], cwd: Optional[str] = None) -> None:
  print("    $ " + " ".join(cmd), flush=True)
  subprocess.run(cmd, cwd=cwd, check=True)


def _confirm(question: str, assume_yes: bool) -> bool:
  if assume_yes:
    return True
  try:
    return input(f"{question} [Y/n] ").strip().lower() in ("", "y", "yes")
  except EOFError:
    return False


def _probe(expect_backend: str = "") -> bool:
  """Compiles a probe theorem; with `expect_backend`, that backend must answer.

  The verifier falls back from a URL to local Lean, so a `--kimina` setup is
  checked with `expect_backend="kimina"`: a local answer would hide a URL that
  does not work.
  """
  import lean_verifier_tool as lv  # pylint: disable=g-import-not-at-top
  _say("Checking the verifier with a probe theorem...")
  try:
    out = lv.run_lean_verifier(lv.LeanVerifierInput(
        code="import Mathlib\ntheorem leap_probe : (2:ℝ) + 2 = 4 := by norm_num",
        target_name="leap_probe", timeout=900))
  except (NotImplementedError, ValueError) as e:
    print(f"    failed: {e}")
    return False
  if expect_backend and out.backend != expect_backend:
    print(f"    failed: the {expect_backend} verifier gave no verdict"
          f" ({out.remote_error or out.error_message}); local Lean is used"
          " while it does not answer")
    return False
  if out.is_valid and out.used_axioms is not None:
    print(f"    ok -- compiled, axioms {out.used_axioms}")
    return True
  print(f"    failed: {out.error_message or 'axiom list missing from output'}")
  return False


def _write_mathlib_project(project: str) -> None:
  """Writes a minimal Lake project pinned to config.LEAN_VERSION.

  `lake new <name> math` would follow Mathlib master, whose toolchain moves on
  every Lean release, so the files are written here instead.
  """
  os.makedirs(project, exist_ok=True)
  files = {
      "lean-toolchain": config.LEAN_TOOLCHAIN + "\n",
      "lakefile.toml": (
          'name = "leap_lean"\n'
          'defaultTargets = ["LeapLean"]\n\n'
          "[[require]]\n"
          'name = "mathlib"\n'
          'scope = "leanprover-community"\n'
          f'rev = "{config.LEAN_VERSION}"\n\n'
          "[[lean_lib]]\n"
          'name = "LeapLean"\n'),
      "LeapLean.lean": "import Mathlib\n",
  }
  for name, content in files.items():
    with open(os.path.join(project, name), "w", encoding="utf-8") as f:
      f.write(content)


# --------------------------------------------------------------------------
# leap lean: make `lean` + Mathlib work in any folder
# --------------------------------------------------------------------------

_PROFILE_MARK = "# Added by `leap lean`: Lean + Mathlib in any folder"


def _project_lean_path(lake: str, project: str) -> str:
  """LEAN_PATH as `lake env` sets it in `project`, without inherited entries."""
  env = {k: v for k, v in os.environ.items() if k != "LEAN_PATH"}
  out = subprocess.run(
      [lake, "env", sys.executable, "-c",
       "import os; print(os.environ['LEAN_PATH'])"],
      cwd=project, env=env, capture_output=True, text=True, encoding="utf-8",
      check=True).stdout
  return out.strip().splitlines()[-1]


def _shell_profiles() -> List[str]:
  home = os.path.expanduser("~")
  found = [os.path.join(home, f) for f in (".bashrc", ".zshrc", ".profile")
           if os.path.exists(os.path.join(home, f))]
  return found or [os.path.join(home, ".profile")]


def _persist_lean_path_posix(lean_path: str) -> List[str]:
  """Writes $LEAP_HOME/env and sources it from the shell profiles."""
  env_file = os.path.join(config.leap_home(), "env")
  os.makedirs(config.leap_home(), exist_ok=True)
  with open(env_file, "w", encoding="utf-8") as f:
    f.write(f"{_PROFILE_MARK}\nexport LEAN_PATH=\"{lean_path}\"\n")
  source = f'[ -f "{env_file}" ] && . "{env_file}"'
  changed = []
  for profile in _shell_profiles():
    try:
      with open(profile, encoding="utf-8") as f:
        text = f.read()
    except OSError:
      text = ""
    if "LEAN_PATH=" in text:
      print(f"    note: {profile} already sets LEAN_PATH; the line leap adds"
            " comes later, so it wins")
    if source in text:
      continue
    with open(profile, "a", encoding="utf-8") as f:
      f.write(f"{'' if not text or text.endswith(chr(10)) else chr(10)}"
              f"\n{_PROFILE_MARK}\n{source}\n")
    changed.append(profile)
  return [env_file] + changed


def _persist_lean_path_windows(lean_path: str) -> List[str]:
  """Sets LEAN_PATH in the user environment (HKCU\\Environment).

  Not `setx`: it truncates values at 1024 characters, and a Mathlib LEAN_PATH
  is longer than that.
  """
  import ctypes  # pylint: disable=g-import-not-at-top
  import winreg  # pylint: disable=g-import-not-at-top
  with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                      winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE) as key:
    try:
      old = winreg.QueryValueEx(key, "LEAN_PATH")[0]
    except OSError:
      old = ""
    if old and old != lean_path:
      print(f"    note: replacing your previous user LEAN_PATH ({old[:80]}...)")
    winreg.SetValueEx(key, "LEAN_PATH", 0, winreg.REG_SZ, lean_path)
  # Tell running programs (Explorer, new terminals) the environment changed.
  ctypes.windll.user32.SendMessageTimeoutW(
      0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, None)
  return [r"HKEY_CURRENT_USER\Environment\LEAN_PATH"]


def _probe_any_folder(lean_path: str) -> bool:
  """Compiles a Mathlib probe with plain `lean` from a folder outside the project."""
  lean = shutil.which("lean")
  if lean is None:
    print("    failed: `lean` not found on PATH")
    return False
  with tempfile.TemporaryDirectory() as tmp:
    src = os.path.join(tmp, "leap_probe.lean")
    with open(src, "w", encoding="utf-8") as f:
      f.write("import Mathlib\ntheorem leap_probe : (2:ℝ) + 2 = 4 := by norm_num\n")
    res = subprocess.run([lean, src], cwd=tmp, capture_output=True, text=True,
                         encoding="utf-8", errors="replace",
                         env=dict(os.environ, LEAN_PATH=lean_path), check=False)
  if res.returncode == 0:
    print("    ok -- `lean` compiled a Mathlib probe outside the project")
    return True
  print(f"    failed: {(res.stdout + res.stderr).strip()[:300]}")
  return False


def _setup_any_folder(lake: str, project: str, assume_yes: bool) -> bool:
  """Points elan's default toolchain and LEAN_PATH at `project`."""
  toolchain = config.project_toolchain(project)
  where = ("your user environment" if os.name == "nt"
           else ", ".join(_shell_profiles()))
  if not _confirm(f"Make `lean` with Mathlib work in any folder? This sets elan's"
                  f" default toolchain to {toolchain} and LEAN_PATH in {where}.",
                  assume_yes):
    print("    skipped; `lake env lean` still works inside " + project)
    return True
  _say(f"Making Lean {toolchain} + Mathlib the default outside projects")
  elan = shutil.which("elan")
  if elan is None:
    print("    failed: `elan` not found on PATH")
    return False
  _run([elan, "default", toolchain])
  lean_path = _project_lean_path(lake, project)
  persist = (_persist_lean_path_windows if os.name == "nt"
             else _persist_lean_path_posix)
  for p in persist(lean_path):
    print(f"    wrote {p}")
  print("    open a new terminal (or `source` your profile) to pick it up")
  return _probe_any_folder(lean_path)


def cmd_lean(args) -> int:
  config.apply()
  if args.kimina:
    path = config.save({
        "LEAN_VERIFIER_BACKEND": "kimina",
        "LEAN_VERIFIER_URL": args.kimina.rstrip("/"),
        "LEAN_VERIFIER_API_KEY": args.api_key or "",
    })
    os.environ.update(LEAN_VERIFIER_BACKEND="kimina",
                      LEAN_VERIFIER_URL=args.kimina.rstrip("/"))
    if args.api_key:
      os.environ["LEAN_VERIFIER_API_KEY"] = args.api_key
    _say(f"Saved kimina settings to {path}")
    return 0 if _probe(expect_backend="kimina") else 1

  # 1. elan / lake
  lake = shutil.which("lake")
  if lake is None:
    if not _confirm("Lean is not installed. Install it with elan "
                    "(the official Lean installer)?", args.yes):
      print("Install elan yourself (https://lean-lang.org/install) and re-run.")
      return 1
    _say("Installing elan")
    if os.name == "nt":
      subprocess.run(ELAN_INSTALL_WINDOWS, check=True)
    else:
      subprocess.run(ELAN_INSTALL, shell=True, check=True)
    config.apply()  # puts ~/.elan/bin on PATH for this process
    lake = shutil.which("lake")
    if lake is None:
      print("elan installed, but `lake` was not found in ~/.elan/bin.")
      return 1

  # 2. a Lake project that depends on Mathlib
  project = os.path.abspath(os.path.expanduser(
      args.project or os.path.join(config.leap_home(), "leap_lean")))
  has_lakefile = any(os.path.exists(os.path.join(project, f))
                     for f in ("lakefile.lean", "lakefile.toml"))
  if not has_lakefile:
    if args.project:
      print(f"{project} has no lakefile. Point --project at a Lake project that"
            " depends on Mathlib, or omit it to let leap create one.")
      return 1
    _say(f"Creating a Mathlib project at {project} (Lean {config.LEAN_VERSION})")
    _write_mathlib_project(project)
    # elan installs the pinned toolchain on first use; this resolves Mathlib.
    _run([lake, "update"], cwd=project)
  else:
    toolchain = config.project_toolchain(project)
    if toolchain != config.LEAN_TOOLCHAIN:
      if not args.project:
        print(f"{project} uses {toolchain or 'no pinned toolchain'}, but LEAP"
              f" needs {config.LEAN_TOOLCHAIN}. Remove that directory and re-run"
              " `leap lean`, or pass --project to keep using it.")
        return 1
      print(f"    warning: {project} uses {toolchain or 'no pinned toolchain'};"
            f" LEAP is tested with {config.LEAN_TOOLCHAIN}.")

  # 3. prebuilt Mathlib binaries, then build the project
  _say("Downloading prebuilt Mathlib (~7 GB, first time only)")
  _run([lake, "exe", "cache", "get"], cwd=project)
  _say("Building")
  _run([lake, "build"], cwd=project)

  # Local Lean is the default, so no backend is saved: a URL the user sets
  # later is then tried first, and this project stays the fallback.
  path = config.save({
      "LEAN_VERIFIER_BACKEND": "",
      "LEAN_PROJECT_DIR": project,
      "LEAN_VERIFIER_URL": "",
      "LEAN_VERIFIER_API_KEY": "",
  })
  os.environ.update(LEAN_VERIFIER_BACKEND="local", LEAN_PROJECT_DIR=project)
  _say(f"Saved settings to {path}")
  if not _probe():
    return 1
  if args.no_global:
    return 0
  return 0 if _setup_any_folder(lake, project, args.yes) else 1


# --------------------------------------------------------------------------
# leap init / doctor / loop / config
# --------------------------------------------------------------------------

_RUN_HINT = {
    "claude": 'claude --agent orchestrator-lemma "Prove problems/foo.lean"',
    "codex": 'codex "$leap-prove Prove problems/foo.lean"',
    "antigravity": "open this folder in Antigravity, choose the "
                   "orchestrator_lemma agent, and ask it to prove problems/foo.lean",
}


def cmd_init(args) -> int:
  import install  # pylint: disable=g-import-not-at-top
  dest = os.path.abspath(os.path.expanduser(args.dest))
  _say(f"Installing LEAP agents for {args.host} into {dest}")
  install.INSTALLERS[args.host](dest)
  os.makedirs(os.path.join(dest, "problems"), exist_ok=True)

  config.apply()
  import lean_verifier_tool as lv  # pylint: disable=g-import-not-at-top
  if not (lv._local_project()  # pylint: disable=protected-access
          or os.environ.get("LEAN_VERIFIER_URL")):
    print("\nNo Lean verifier configured yet -- run `leap lean` once.")
  missing = [b for b in install.LEAP_BINS if shutil.which(b) is None]
  if missing:
    print(f"\n{', '.join(missing)} not on PATH. Install LEAP with pipx/uv"
          " (see README) or activate the virtualenv you installed it into.")
  print("\nReady. Put a .lean file in problems/ and run:\n  " + _RUN_HINT[args.host])
  return 0


def cmd_doctor(args) -> int:
  config.apply()
  import leap_doctor  # pylint: disable=g-import-not-at-top
  return leap_doctor.main(["leap doctor"] + (["--offline"] if args.offline else []))


def cmd_loop(args) -> int:
  """Runs safe_lemma_runner.sh with host-appropriate defaults."""
  if not args.cli_args:
    print("usage: leap loop <claude|codex|...> <args for that CLI>")
    return 2
  config.apply()
  cli, rest = args.cli_args[0], args.cli_args[1:]
  env = dict(os.environ, LEAP_AGENT_CLI=cli, LEAP_PYTHON=sys.executable)
  if os.path.basename(cli) == "codex":
    env.setdefault("LEAP_RESUME_FLAG", "")
    env.setdefault("LEAP_RESUME_PREFIX", "$leap-prove ")
  if args.root:
    env["LEAP_TARGET_DIR"] = args.root
  runner = os.path.join(PKG_DIR, "safe_lemma_runner.sh")
  return subprocess.call(["bash", runner, *rest], env=env)


def cmd_config(_args) -> int:
  print(f"# {config.config_path()}")
  saved = config.load()
  for k in config.KNOWN_KEYS:
    v = saved.get(k, "")
    if k.endswith("API_KEY") and v:
      v = v[:4] + "…"
    shown = os.environ.get(k)
    note = "  (overridden by environment)" if shown and shown != saved.get(k) else ""
    print(f"{k}={v}{note}")
  return 0


def main(argv: Optional[List[str]] = None) -> int:
  parser = argparse.ArgumentParser(
      prog="leap", description="LEAP: multi-agent Lean 4 theorem proving.")
  sub = parser.add_subparsers(dest="cmd", required=True)

  p = sub.add_parser(
      "lean", help=f"one-time Lean {config.LEAN_VERSION} + Mathlib setup")
  p.add_argument("--project", help="use an existing Lake project with Mathlib")
  p.add_argument("--kimina", metavar="URL",
                 help="use a kimina-lean-server while it answers (local Lean otherwise)")
  p.add_argument("--api-key", help="kimina API key (LEAN_SERVER_API_KEY)")
  p.add_argument("--no-global", action="store_true",
                 help="don't set elan's default toolchain and LEAN_PATH, so"
                      " Mathlib works only inside the project")
  p.add_argument("-y", "--yes", action="store_true", help="don't ask")
  p.set_defaults(fn=cmd_lean)

  if HAS_AGENTS:
    p = sub.add_parser("init", help="install the agents for one CLI")
    p.add_argument("host", choices=HOSTS)
    p.add_argument("--dest", default=".", help="workspace (default: here)")
    p.set_defaults(fn=cmd_init)

  p = sub.add_parser("doctor", help="check the setup")
  p.add_argument("--offline", action="store_true")
  p.set_defaults(fn=cmd_doctor)

  if HAS_RUNNER:
    p = sub.add_parser("loop", help="rerun an agent CLI until the proof is done")
    p.add_argument("--root", help="problem root to watch (default: from prompt)")
    p.add_argument("cli_args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_loop)

  p = sub.add_parser("config", help="show saved settings")
  p.set_defaults(fn=cmd_config)

  args = parser.parse_args(argv)
  try:
    return args.fn(args)
  except subprocess.CalledProcessError as e:
    print(f"\ncommand failed ({e.returncode}): {' '.join(map(str, e.cmd))}")
    return 1
  except KeyboardInterrupt:
    return 130


if __name__ == "__main__":
  sys.exit(main())
