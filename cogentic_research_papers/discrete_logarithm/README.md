# Discrete Logarithms in Finite Fields of Fixed Characteristic in Sub-Quasi-Polynomial Time

[This paper](paper.pdf) gives an unconditional classical zero-error Las Vegas
algorithm for the Discrete Logarithm Problem in the multiplicative group
$\mathbb F_{p^n}^\times$ of any finite field of small characteristic: on every
prime $p \ge 2$, integer $n \ge 1$, degree $n$ monic irreducible
$f \in \mathbb F_p[X]$ presenting $\mathbb F_{p^n} = \mathbb F_p[X]/(f(X))$,
and nonzero $g, h \in \mathbb F_{p^n}^\times$, it either returns an integer
$x \in \lbrace 0, \dots, p^n - 2\rbrace$ with $g^x = h$ or returns an explicit
exponent $E \in \lbrace 1, \dots, p^n - 1\rbrace$ certifying
$h \notin \langle g \rangle$ (with $g^E = 1$ and $h^E \ne 1$, without factoring
$p^n - 1$), in worst-case expected bit complexity

$$
\exp\left(\frac{2 + o(1)}{\ln\ln\ln n}\ (\ln n)^2\right) = (pn)^{o(\log n)}
$$

for every fixed prime $p \ge 2$ as $n \to \infty$ (and uniformly whenever
$\ln p \le \frac{1}{4}(\ln\ln\ln n)^2$), and
$\exp(o((\ln n)^2)) = (pn)^{o(\log n)}$ along every sequence with
$n \to \infty$ and $(\ln p)/(\ln\ln n) \to 0$. A complete, unconditional,
Mathlib-only Lean 4 (`v4.35.0`) formalization of the entire proof has also been
completed and will be released in a companion submission.

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
