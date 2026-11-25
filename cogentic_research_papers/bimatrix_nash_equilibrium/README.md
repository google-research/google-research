# An Exact Polynomial-Time Algorithm for $1/5$-Approximate Nash Equilibria in Bimatrix Games

[This paper](paper.pdf) constructs an explicit deterministic algorithm,
`MaximinKiteNE`, that, given any integers $m, n \ge 1$ and any
two-player normal-form (bimatrix) game $G = (R, C)$ with rational payoff
matrices $R, C \in ([0, 1] \cap \mathbb{Q})^{m \times n}$ of total binary
encoding length $L$, runs in deterministic time $\mathrm{poly}(L)$ and
outputs a rational mixed strategy profile $(x^{\mathrm{out}}, y^{\mathrm{out}})$
in $(\Delta_m \cap \mathbb{Q}^m) \times (\Delta_n \cap \mathbb{Q}^n)$
of binary encoding length $\mathrm{poly}(L)$ satisfying maximum unilateral
regret

$$f(x^{\mathrm{out}}, y^{\mathrm{out}}) \le \frac{1}{5}.$$

This establishes an exact polynomial-time $r$-approximation ($\delta = 0$) at
$r = 1/5$—improving the earlier $1/3 + \delta$ barrier (Deligkas et al., 2023)
and the recent bound of approximately $0.30954 + \delta$ (Li and Li, 2026)—and
immediately yields a deterministic polynomial-time approximation scheme (PTAS)
at threshold $r = 1/5$ computing a rational $(1/5 + \delta)$-approximate Nash
equilibrium for every rational $\delta > 0$ in time
$\mathrm{poly}(L) \le \mathrm{poly}(L, 1/\delta)$. A complete,
unconditional, Mathlib-only Lean 4 (`v4.35.0`) formalization of the entire proof
has also been completed and will be released in a companion submission.

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
