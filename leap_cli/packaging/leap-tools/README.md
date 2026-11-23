# leap-tools

A pip package with only the three CLI tools the LEAP agents call, plus the
`leap lean`, `leap doctor` and `leap config` setup commands. It leaves out
`leap init`, `leap loop`, the agent prompts and the skills:

| Command | Module |
| :-- | :-- |
| `state_manager_bin` | `leap/state_manager.py` (proof graph + phase machine) |
| `lean_verifier_tool_bin` | `leap/lean_verifier_tool.py` (local Lean; a set kimina URL while it answers) |
| `lean_search_tool_bin` | `leap/lean_search_tool.py` (leansearch.net / Loogle) |

```bash
pip install "leap-tools @ git+https://github.com/google-research/google-research.git#subdirectory=leap_cli/packaging/leap-tools"
leap lean      # one-time: Lean v4.34.0 + matching Mathlib (~11 GB on disk, 10–20 min)
leap doctor    # optional: check the setup end to end
```

To make the install reproducible, pin it to a full commit SHA of
`google-research/google-research` by adding `@<sha>` after `.git`. A full
commit SHA cannot be moved the way a branch can. To upgrade, review the
changes under `leap_cli/` and replace the SHA:
`git log <sha>..origin/master -- leap_cli/leap leap_cli/packaging/leap-tools`.

A wheel cannot run code at install time, so Lean comes from `leap lean`, not
from `pip install`. It installs elan if missing, then creates
`~/.leap/leap_lean` pinned to `leanprover/lean4:v4.34.0` with Mathlib at its
`v4.34.0` tag, and saves the settings.

Finally, after asking (`-y` skips the question, `--no-global` skips the step),
it makes plain `lean file.lean` see Mathlib in any folder:

- `elan default leanprover/lean4:v4.34.0`, so `lean` outside a project is
  v4.34.0. Projects with their own `lean-toolchain` are unaffected.
- `LEAN_PATH` set to the project's library paths. On Linux/macOS it is written
  to `~/.leap/env`, which `~/.bashrc`, `~/.zshrc` and/or `~/.profile` source.
  On Windows it is set in the user environment (`HKCU\Environment`). Open a new
  terminal to pick it up.

To undo: `elan default <previous toolchain>`, and remove the `~/.leap/env`
line from your profile (Windows: delete the user `LEAN_PATH` variable).

The wheel is built from the same sources in `../../leap/` via hatchling
`force-include`, so nothing is duplicated. Build wheels only; an sdist would
not contain the files:

```bash
uv build --wheel packaging/leap-tools   # or: python -m build --wheel packaging/leap-tools
```

The verifier uses local Lean by default: `LEAN_PROJECT_DIR`, else the
`~/.leap/leap_lean` project `leap lean` creates. To use an existing project,
run `leap lean --project DIR` (or set `LEAN_PROJECT_DIR`). A verifier URL is
the only exception: with `leap lean --kimina URL [--api-key K]` (or
`LEAN_VERIFIER_URL`, + `LEAN_VERIFIER_API_KEY`) set, the URL is tried first,
and whenever it gives no verdict the check runs on local Lean.
`LEAN_VERIFIER_BACKEND=local` never calls a URL.

Do not install it next to `leap-cli`; both provide the `leap` package and the
same commands.
