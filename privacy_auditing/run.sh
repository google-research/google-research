#!/bin/bash
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


# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# run.sh — Minimal smoke test for privacy_auditing.
#
# Runs lightweight unit tests that require only CPU (numpy, scipy, absl-py).
# These tests do not require GPU/TPU, model checkpoints, or dataset files.
#
# Usage (from the google-research/ root):
#   bash privacy_auditing/run.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "=== Privacy Auditing — Smoke Tests ==="
echo "Running from: ${REPO_ROOT}"
echo ""

# --------------------------
# 1. Privacy metrics (GDP upper bound with interval).
# --------------------------
echo "--- Test 1: Privacy metrics (GDP upper bound) ---"
python -c "
import numpy as np
from privacy_auditing import metrics

rng = np.random.default_rng(seed=42)
# Simulate audit scores: 'in' scores slightly higher than 'out' scores.
in_scores = rng.normal(loc=0.5, scale=1.0, size=5000)
out_scores = rng.normal(loc=0.0, scale=1.0, size=5000)

mu_low, mu_hat, mu_high = metrics.audit_gdp(
    in_scores, out_scores, alpha=0.9, central_with_interval=True, seed=42
)
print(f'  GDP mu estimate: {mu_hat:.4f} (90% CI: [{mu_low:.4f}, {mu_high:.4f}])')
assert mu_low <= mu_hat <= mu_high, 'Confidence interval ordering violated'
assert mu_hat > 0, 'Expected positive mu for separated scores'
print('PASS: Privacy metrics')
"
echo ""

# --------------------------
# 2. Privacy metrics (GDP point estimate).
# --------------------------
echo "--- Test 2: GDP upper bound (point estimate) ---"
python -c "
import numpy as np
from privacy_auditing import metrics

rng = np.random.default_rng(seed=42)
in_scores = rng.normal(loc=1.0, scale=1.0, size=3000)
out_scores = rng.normal(loc=0.0, scale=1.0, size=3000)

mu_hat = metrics.audit_gdp(in_scores, out_scores, seed=42)
print(f'  GDP mu estimate: {mu_hat:.4f}')
assert mu_hat > 0, 'Expected positive mu for separated scores'
print('PASS: GDP upper bound')
"
echo ""

echo "=== All smoke tests passed ==="
echo ""
echo "To run a full training job (requires GPU/TPU + dataset):"
echo "  # Set dataset environment variables:"
echo "  export ENRON_TRAIN_PATH=/path/to/enron/train.bagz"
echo "  export ENRON_TEST_PATH=/path/to/enron/test.bagz"
echo "  export GEMMA3_TOKENIZER_PATH=/path/to/gemma3/tokenizer.model"
echo ""
echo "  # Run training (from google-research/ root):"
echo "  python -m privacy_auditing.train \\"
echo "    --model_version=gemma3_1b \\"
echo "    --work_dir=/tmp/privacy_auditing_output \\"
echo "    --max_steps=10"
