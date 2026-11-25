# Unconditional Integer Factorization in Time $L_N[1/3, O(1)]$

[This paper](paper.pdf) constructs an unconditional, zero-error Las Vegas
algorithm on a single fixed uniform classical multitape Turing machine
$\mathcal{M}$ that computes the exact complete prime factorization
$N = \prod_{1 \le i \le k} p_i^{e_i}$ ($p_1 < \dots < p_k$ prime, $e_i \ge 1$)
of every integer $N \ge 2$ given in standard binary representation, halting
with probability $1$ over its read-once fair coin flips, in worst-case expected
time $L_N[1/3, 105]$ and worst-case deterministic workspace $L_N[1/3, 40]$,
where

$$L_x(\alpha, c) := \exp\!\bigl(c\,(\ln x)^\alpha (\ln\ln x)^{1-\alpha}\bigr)$$

and $T(N) \le L_N[\alpha, c]$ means

$$
\limsup_{N \to \infty}
\frac{\ln \max(1, T(N))}{(\ln N)^\alpha (\ln\ln N)^{1-\alpha}} \le c.
$$

Every prime factor in the output is certified deterministically via
Agrawal–Kayal–Saxena primality testing and direct multiplication against $N$,
with effectively computable constants and no unproven number-theoretic
hypotheses. A complete, unconditional, Mathlib-only Lean 4 (`v4.35.0`)
formalization of the entire proof has also been completed and will be released
in a companion submission.

This paper and its Lean 4 formalization were entirely obtained by Cogentic
(arXiv preprint [arXiv:2609.40324](https://arxiv.org/abs/2609.40324)), an
agentic framework for mathematical discovery, using Gemini 4 as the base model.
There was no human interaction other than the initial prompt for Cogentic to
generate a full proof and formalization.
The Cogentic harness was built by Yang Cai, Vineet Gupta, Yanchen Jiang,
Christopher Liaw, Aranyak Mehta, Grigoris Velegkas, and Di Wang (Google Research
and Google DeepMind).

## License and disclaimer

Copyright 2026 Google LLC

The paper in this directory is licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/legalcode)
(CC-BY-4.0).

Unless required by applicable law or agreed to in writing, all materials
distributed here under the CC-BY-4.0 license are distributed on an "AS IS"
BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the license for the specific language governing permissions and limitations
under the license.

This is not an official Google product.
