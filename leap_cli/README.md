# LEAP CLI

LEAP proves Lean 4 theorems with a team of coding agents, inside the agent CLI
you already use: **Claude Code**, **Codex**, or **Antigravity**.
It is the CLI-agent version of [arXiv:2606.03303](https://arxiv.org/abs/2606.03303).

## Quickstart

```bash
pipx install "git+https://github.com/google-research/google-research.git#subdirectory=leap_cli"   # or: uv tool install / pip install
leap lean                 # one-time: installs Lean v4.34.0 + Mathlib (~11 GB on disk, 10–20 min)

mkdir my-proofs && cd my-proofs
leap init claude          # or: codex / antigravity
```

Put a Lean file in `problems/`. It should contain the imports, any
definitions, and the theorem with `sorry` as its proof:

```lean
import Mathlib

theorem amgm2 (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    Real.sqrt (a * b) ≤ (a + b) / 2 := by
  sorry
```

Then start the orchestrator:

| CLI | Command |
| :-- | :-- |
| Claude Code | `claude --agent orchestrator-lemma "Prove problems/amgm2.lean"` |
| Codex | `codex '$leap-prove Prove problems/amgm2.lean'` |
| Antigravity | Open the folder, choose the `orchestrator_lemma` agent, and ask it to prove `problems/amgm2.lean`. |

The result is written to `runs/amgm2/assembled/final_proof.lean`. That file
contains no `sorry`, compiles in the Lean kernel, and uses only the standard
axioms.

## Commands

| Command | What it does |
| :-- | :-- |
| `leap lean` | Installs [elan](https://github.com/leanprover/elan) if missing, creates a Mathlib project at `~/.leap/leap_lean` pinned to Lean v4.34.0 (Mathlib at its `v4.34.0` tag), downloads the prebuilt Mathlib cache, and saves the settings. It then (after asking) makes `lean` + Mathlib work in any folder: it sets elan's default toolchain and `LEAN_PATH` (in your shell profile, or the user environment on Windows). Pass `--project DIR` to use an existing Mathlib project instead, or `--no-global` to skip the any-folder step. |
| `leap lean --kimina URL [--api-key K]` | Uses a [kimina-lean-server](https://github.com/project-numina/kimina-lean-server) while it answers; when it gives no verdict, the check runs on local Lean (`LEAN_PROJECT_DIR`, else `~/.leap/leap_lean`). It is faster when many workers run in parallel. Local Lean is the default whenever no URL is set. |
| `leap init <cli>` | Installs the agents and skills for `claude`, `codex` or `antigravity` into the current folder. |
| `leap doctor` | Checks the whole setup, including compiling a probe theorem. |
| `leap loop <cli> ...` | Reruns a headless CLI until the proof is done, resuming after crashes and rate limits. Example: `leap loop claude --agent orchestrator-lemma -p "Prove problems/amgm2.lean"`. |
| `leap config` | Shows the saved settings in `~/.leap/config.json`. |

Settings are saved once, so nothing has to be exported. Environment variables
with the same names override the saved values. See
[docs/configuration.md](docs/configuration.md).

## How it works

```
orchestrator ── claim_next ─▶ state_manager_bin ◀── submit / declare ── worker agents
     │                        (proof graph, phase                    informal prover
     └── dispatches ──────▶    machine, soundness gates)             formal prover
                                       │                             decomposition reviewer
                    lean_verifier_tool_bin   lean_search_tool_bin    final verifier
                    (local Lean / kimina)    (leansearch.net / Loogle)
```

1. The orchestrator splits the theorem into a graph of subgoals and sends each
   node to the agent its phase needs. The phases are: plan it in prose, prove
   or split it in Lean, review the split, or refute it.
2. Agents never edit the proof state directly. Each turn ends with one
   `state_manager_bin` command.
3. A submission is accepted only after Lean compiles it and `#print axioms`
   shows nothing beyond `propext`, `Classical.choice` and `Quot.sound`.
4. A run ends in one of two ways: an assembled proof, or a kernel-checked
   disproof. It has no give-up condition.

See [docs/architecture.md](docs/architecture.md) for the full design.

## Good to know

- **Cost.** A hard problem can take hundreds of subagent runs. Start with
  something small.
- **Mathlib version.** Write problems against the Mathlib version that
  `leap lean` installed.
- **Search.** [leansearch.net](https://leansearch.net) is a free community
  service. When it is down, LEAP falls back to
  [Loogle](https://loogle.lean-lang.org) automatically.
- **Platforms.** LEAP runs on macOS and Linux. On Windows, use WSL.

## Development

```bash
# google-research is a large monorepo; fetch only this directory.
git clone --depth=1 --filter=blob:none --sparse https://github.com/google-research/google-research.git
cd google-research && git sparse-checkout set leap_cli && cd leap_cli
pip install -e ".[test]"
python3 -m unittest discover -s tests -p '*_test.py'
```

```
leap/          Python package: state manager, proof graph, verifier/search clients, CLI
leap/agents/   host-neutral agent prompts, rendered per CLI by `leap init`
leap/skills/   Mathlib search and tactics skills
tests/         offline test suite (Lean and web services are stubbed)
docs/          architecture, configuration, testing
```

## Authors

LEAP CLI is written by Po-Nien Kung and Linfeng Song. The LEAP paper is by
Po-Nien Kung, Linfeng Song, Dawsen Hwang, Jinsung Yoon, Chun-Liang Li, Simone
Severini, Mirek Olšák, Edward Lockhart, Quoc V. Le, Burak Gokturk, Thang Luong,
Tomas Pfister and Nanyun Peng.

## Citation

If you use LEAP, please cite:

```bibtex
@article{kung2026leap,
  title   = {{LEAP}: Supercharging {LLMs} for Formal Mathematics with Agentic Frameworks},
  author  = {Kung, Po-Nien and Song, Linfeng and Hwang, Dawsen and Yoon, Jinsung and
             Li, Chun-Liang and Severini, Simone and Ol{\v{s}}{\'a}k, Mirek and
             Lockhart, Edward and Le, Quoc V. and Gokturk, Burak and Luong, Thang and
             Pfister, Tomas and Peng, Nanyun},
  journal = {arXiv preprint arXiv:2606.03303},
  year    = {2026}
}
```

## License

Apache 2.0

*Disclaimer: This is not an officially supported Google product. This project
is not eligible for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).*
