# Breaking the Square-Root Barrier for the Log-Rank Conjecture

[This paper](paper.pdf) proves a polynomial
improvement over Lovett's $O(\sqrt{r} \log r)$ bound for the Lovász–Saks
log-rank conjecture. Setting $\varepsilon := 5 \times 10^{-6}$, $C := 2^{38}$,
and $C' := 750\ C$, every Boolean matrix $M \in \lbrace0, 1\rbrace^{X \times Y}$
of real rank $r = \mathrm{rank}_{\mathbb{R}}(M) \ge 1$ over finite nonempty sets
$X, Y$ satisfies:

1. There exists an $M$-monochromatic combinatorial rectangle
   $A \times B \subseteq X \times Y$ of density
   $$\frac{|A| \cdot |B|}{|X| \cdot |Y|} \ge 2^{-C\ r^{1/2 - \varepsilon}}.$$
2. Whenever $r \ge 2$, the monochromatic partition number $\chi(M)$ and
   deterministic two-party communication complexity $D^{\mathrm{cc}}(M)$ satisfy
   $$\log_2 \chi(M) \le D^{\mathrm{cc}}(M) \le C'\ r^{1/2 - \varepsilon}.$$

The proof reduces monochromatic rectangle density via Gaussian hyperplane
rounding and box-constrained biasing to refuting the existence of biased
Kullback–Leibler-stable sign matrices $S \in \lbrace-1, +1\rbrace^{X \times Y}$
of rank at most $r + 1$ above $R_0 := \lceil e^{4 \times 10^6} \rceil$
(Theorem 3.1).
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
and Google DeepMind; Yang Cai is also affiliated with Yale University).

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
