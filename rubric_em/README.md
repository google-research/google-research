# RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards

Reference implementation of the core components of *RubricEM: Meta-RL with
Rubric-guided Policy Decomposition beyond Verifiable Rewards*
([arXiv:2605.10899](https://arxiv.org/abs/2605.10899)).

Authors: Gaotang Li, Bhavana Dalvi
Mishra, Zifeng Wang, Jun Yan, Yanfei Chen, Chun-Liang Li, Long T. Le, Rujun Han,
George Lee, Hanghang Tong, Chen-Yu Lee, Tomas Pfister.

> **Important:** this directory contains only the RubricEM code written by the
> authors. It is **not runnable on its own**. RubricEM is built on top of
> [DR Tulu](https://github.com/rlresearch/dr-tulu) (Apache-2.0), which you
> install separately. The RubricEM training entry point (an SS-GRPO variant of
> DR Tulu's `open_instruct/grpo_fast.py`), RubricEM's changes to DR Tulu's
> search tools, inference workflow and rubric reward, and the rubric prompts are
> **not included**; see "Integrating with DR Tulu" below.

## What's here

| Path                                                       | What it is      |
| ---------------------------------------------------------- | --------------- |
| `rl/open-instruct/open_instruct/rubric_bank.py`            | Rubric bank     |
:                                                            : data structure  :
:                                                            : and similarity  :
:                                                            : retrieval       :
| `rl/open-instruct/open_instruct/rubric_bank_generation.py` | Reflection      |
:                                                            : generation and  :
:                                                            : judge scoring   :
:                                                            : for rubric-bank :
:                                                            : items           :
| `rl/open-instruct/open_instruct/rubric_bank_injection.py`  | Injecting       |
:                                                            : retrieved       :
:                                                            : rubric-bank     :
:                                                            : items into      :
:                                                            : prompts         :
| `rl/open-instruct/open_instruct/stage_scoring.py`          | Stage-wise      |
:                                                            : rubric          :
:                                                            : generation and  :
:                                                            : scoring         :
:                                                            : (SS-GRPO        :
:                                                            : rewards)        :
| `rl/open-instruct/open_instruct/stage_utils.py`            | Parsing of the  |
:                                                            : plan / research :
:                                                            : / review /      :
:                                                            : answer stages   :
| `data_generation/`                                         | Step-wise SFT   |
:                                                            : data-generation :
:                                                            : pipeline        :
:                                                            : (Gemini Batch   :
:                                                            : API + Google    :
:                                                            : Search          :
:                                                            : grounding,      :
:                                                            : Semantic        :
:                                                            : Scholar)        :
| `script/`                                                  | Evaluation      |
:                                                            : launcher and    :
:                                                            : per-benchmark   :
:                                                            : eval / citation :
:                                                            : utilities       :
| `agent/workflows/eval_bank_formatter.py`                   | Formats         |
:                                                            : eval-time       :
:                                                            : rubric-bank     :
:                                                            : entries         :
| `eval_trajectories/trajectory_viewer.html`                 | Self-contained  |
:                                                            : browser viewer  :
:                                                            : for eval        :
:                                                            : `answers.jsonl` :
:                                                            : files           :

Files keep the paths they have inside a DR Tulu checkout, so you can copy them
into one directly.

## Integrating with DR Tulu

1.  Clone DR Tulu (the code was developed against commit `9d7b0371c0`) and
    follow its setup instructions.
2.  Copy this directory's `rl/`, `agent/`, `script/` and `data_generation/`
    folders into the DR Tulu checkout.
3.  To train or evaluate full RubricEM, you also need to re-create, following
    the paper:
    *   **SS-GRPO training loop:** in a copy of `open_instruct/grpo_fast.py`,
        compute per-stage advantages with `stage_scoring.py` / `stage_utils.py`
        and add flags such as `--use_stagewise_credit` and `--use_rubric_bank`.
    *   **Rubric-bank injection at inference:** hook `rubric_bank_injection.py`
        into DR Tulu's `agent/workflows/auto_search_sft.py`.
    *   **Search tool and reward changes:** Gemini Google-Search-grounded search
        in `agent/dr_agent/tool_interface/mcp_tools.py`, and the rubric reward
        in `open_instruct/search_rewards/utils/rubric_utils.py`.
    *   **Structured-scaffold rubric prompts** (plan, research, review, answer)
        in `agent/dr_agent/shared_prompts/`.

## API keys

```bash
export GEMINI_API_KEY="..."   # Gemini API: data generation, search tool, judge
export S2_API_KEY="..."       # Optional: Semantic Scholar
```

Use of the Gemini API is subject to the
[Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms).

## Smoke test

```bash
bash rubric_em/run.sh   # byte-compiles the modules; no GPU or API keys needed
```

## Citation

```bibtex
@article{li2026rubricem,
  title={RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards},
  author={Li, Gaotang and Dalvi Mishra, Bhavana and Wang, Zifeng and Yan, Jun and Chen, Yanfei and Li, Chun-Liang and Le, Long T. and Han, Rujun and Lee, George and Tong, Hanghang and Lee, Chen-Yu and Pfister, Tomas},
  journal={arXiv preprint arXiv:2605.10899},
  year={2026}
}
```

This is not an officially supported Google product.
