#!/usr/bin/env bash
# Run the rubric judge over one agent's answers, then aggregate.
#
# Required env vars:
#   QUESTIONS  path to questions.jsonl produced by financegym.curation
#   ANSWERS    path to answers.jsonl produced by an agent
#   AGENT      short agent identifier stamped onto the score records
#   SCORES_OUT where to write the per-question score JSONL
#
# Optional:
#   MODEL      judge model (defaults to a current Gemini alias; override as needed)

set -euo pipefail

QUESTIONS="${QUESTIONS:?set QUESTIONS=...}"
ANSWERS="${ANSWERS:?set ANSWERS=...}"
AGENT="${AGENT:?set AGENT=...}"
SCORES_OUT="${SCORES_OUT:?set SCORES_OUT=...}"
MODEL="${MODEL:-gemini-flash-latest}"
PYTHON="${PYTHON:-python}"

"$PYTHON" - <<PY
import json
from financegym.judge.rubric_judge import judge_pair_to_record

questions = {json.loads(l)["question"]: json.loads(l) for l in open("$QUESTIONS")}
answers = [json.loads(l) for l in open("$ANSWERS")]

with open("$SCORES_OUT", "w") as out:
    for ans in answers:
        q = questions.get(ans.get("question"))
        if q is None:
            continue
        rec = judge_pair_to_record("$AGENT", q, ans, model="$MODEL")
        if rec is not None:
            out.write(json.dumps(rec) + "\n")
PY
