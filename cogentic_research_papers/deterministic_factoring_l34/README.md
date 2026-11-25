# Unconditional Deterministic Subexponential Integer Factoring and Quadratic Nonresidue Construction

[This paper](paper.pdf) constructs explicit,
parameter-free deterministic algorithms for general integer factoring,
$(p - 1)$-multiple order splitting, and computing quadratic nonresidues modulo
odd primes:

1. **General integer factoring (Theorem 1.1):** An explicit deterministic
   algorithm `algFactor` that, given any composite integer $N \ge 4$, outputs a
   nontrivial divisor $1 < d < N$ of $N$ in worst-case bit-operation complexity
   $O\!\bigl(\exp(3\,(\ln N)^{3/4})\bigr) = L_N[3/4, o(1)]$, where
   $$
   L_x(\alpha, c) := \exp\!\bigl(c\,(\ln x)^\alpha (\ln\ln x)^{1-\alpha}\bigr).
   $$
2. **Order splitting and prime quadratic nonresidues (Theorem 1.2):**
   - An explicit deterministic algorithm `algSplit` that, given any composite
     non-perfect-power $N \ge 4$ and any even integer $\widetilde{M} \ge 2$
     divisible by $p - 1$ for every prime factor $p \mid N$, outputs a
     nontrivial divisor $1 < d < N$ of $N$ in worst-case bit-operation
     complexity
     $$
     \exp\!\bigl(O((\ln N)^{1/2} \ln\ln N)\bigr)
     \cdot (\ln \widetilde{M})(\ln N)^2.
     $$
   - An explicit deterministic algorithm `algQnr` that, given any odd prime
     $p \ge 3$, outputs an integer $z \in \{1, \dots, p - 1\}$ with Legendre
     symbol $\bigl(\frac{z}{p}\bigr) = -1$ in worst-case bit-operation
     complexity $\exp\!\bigl(O((\ln p)^{1/2} \ln\ln p)\bigr) = L_p[1/2, O(1)]$.
3. **Deterministic $(p - 1)$-exponent annihilation (Theorem 1.3):** An explicit
   deterministic subroutine `algExp` that, given any odd composite
   non-perfect-power integer $N \ge 9$, runs in worst-case time
   $O\!\bigl(\exp(3\,(\ln N)^{3/4})\bigr)$ and either outputs a nontrivial
   divisor of $N$ directly or outputs an even integer $\widetilde{M} \ge 2$
   with $\ln \widetilde{M} \le \exp\!\bigl((\ln N)^{0.65}\bigr)$ and
   $(p - 1) \mid \widetilde{M}$ for every prime factor $p \mid N$.

A complete, unconditional, Mathlib-only Lean 4 (`v4.33.0`) formalization of the
entire proof has also been completed and will be released in a companion
submission.

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
