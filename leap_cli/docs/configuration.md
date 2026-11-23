# Configuration

`leap lean` saves your verifier settings to `~/.leap/config.json`, and every
LEAP command reads them, so normally there is nothing to configure. Each
setting can be overridden by an environment variable of the same name. Run
`leap config` to see the current values.

## Verifier

| Variable | Set by | Effect |
| :--- | :--- | :--- |
| `LEAN_VERIFIER_BACKEND` | `leap lean --kimina` | Usually unset. Local Lean is the default, and a set URL is tried first while it answers; `local` never calls a URL; `kimina` means the same as unset. |
| `LEAN_PROJECT_DIR` | `leap lean [--project DIR]` | A Lake project that depends on Mathlib. Local Lean runs `lake env lean --json` in it. Unset means `~/.leap/leap_lean` (`$LEAP_HOME/leap_lean`), if it exists. |
| `LEAN_VERIFIER_URL` | `leap lean --kimina URL` | Base URL of a kimina-lean-server, which receives `POST {url}/api/check`. Used while it gives verdicts; otherwise the check runs on local Lean. |
| `LEAN_VERIFIER_API_KEY` | `leap lean --api-key K` | Sent as `Authorization: Bearer K`. This matches the server's `LEAN_SERVER_API_KEY`. |
| `LEAN_VERIFIER_MAX_ATTEMPTS` | — | Transport retries for kimina (default 3). |
| `LEMMA_MAX_INFRA_RETRIES` | — | How often a node is retried when verification returned no verdict (default 3). |

Any other service can stand in for kimina if it speaks the same HTTP contract.
`VERIFIER_WIRE_CONTRACT` in
[`leap/lean_verifier_tool.py`](../leap/lean_verifier_tool.py) documents it.
The service must return the output of `#print axioms`, because the soundness
gate depends on it.

### Running kimina-lean-server

```bash
docker run -d --name kimina -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
leap lean --kimina http://localhost:8000
```

## Search

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `LEAN_SEARCH_ENDPOINT` | `https://leansearch.net/search` | Primary search service (natural language). |
| `LOOGLE_ENDPOINT` | `https://loogle.lean-lang.org/json` | Fallback search (names and type patterns). |
| `LEAN_SEARCH_SUBAGENT_QUOTA` | `-1` | Maximum searches per agent id. A negative value means unlimited, and `0` disables search. |
| `LEAP_AGENT_ID` | — | The id the quota is charged to. Under Antigravity, its conversation id is used automatically. |

## Paths

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `LEAP_HOME` | `~/.leap` | Where `config.json` and the default Mathlib project live. |

## What `leap init` writes

| CLI | Files |
| :--- | :--- |
| Claude Code | `.claude/agents/*.md`, `.claude/skills/`, and `.claude/settings.json`. The settings allow the three LEAP tools to run without a prompt and raise the Bash timeout to 15 minutes. |
| Codex | `.codex/agents/*.toml` (the four workers), `.agents/skills/`, and the orchestrator as the `$leap-prove` skill. For 8 parallel workers, set `[agents] max_concurrent_threads_per_session = 8` in `~/.codex/config.toml`. |
| Antigravity | `.agents/agents/*.md` (using Antigravity's own tool names) and `.agents/skills/`. |

The prompts in [`leap/agents/`](../leap/agents) are written once in
host-neutral form. `leap init` translates the tool names for each CLI and adds
a short "Host notes" section. To pick up edits to a prompt, run `leap init`
again.
