# FinanceHarness: Automating Financial Deep Research

A model-agnostic agent for financial deep research: web research (search → visit
→ cite) plus an equity slice (fundamentals, comps, DCF, WACC). Headless core;
`main.py` is the entry point.

## Install

```bash
uv sync            # or: pip install -e .
```

Backbones are configured in [`configs/providers.json`](configs/providers.json)
and `financeharness/providers/`. Set the API key for your chosen backbone via
its environment variable, or point a profile at any OpenAI-compatible endpoint.

## Run

One-shot research — prints the cited report to stdout, progress to stderr:

```bash
python main.py -p "Estimate NVDA's intrinsic value with a DCF."
python main.py -p "Apple's competitive position in 2026?" --mode research
python main.py -p "..." --save run.json                 # persist the full trajectory
echo "What's AAPL's P/E?" | python main.py -p            # question piped via stdin
```

Options: `--mode {auto|research|analytical}`, `--profile NAME`, `--reader NAME`,
`--save PATH`, `--quiet`.

### Verify your install

```bash
python main.py --list        # prints available profiles; confirms a working install
```

### Optional HTTP+SSE service

The package also ships an HTTP+SSE service (for a remote UI or programmatic
use):

```bash
uv run fh serve              # HTTP+SSE on 127.0.0.1:8080
```

## Extend with skills

A **skill** is a `SKILL.md` (YAML frontmatter + markdown body) that orchestrates
the existing tools — drop one in and it's discovered, no code changes. Discovery
precedence: bundled `financeharness/skills/` → project `./skills/` →
`FH_SKILLS_DIR`.

## License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
International License (CC BY-NC 4.0).

*Non-Commercial Use Only:* This tool, code, and benchmark datasets are provided
strictly for academic and non-commercial research purposes. Commercial use,
including integration into commercial financial advisory services or using
generated outputs to provide commercial investment advice, is strictly
prohibited.

