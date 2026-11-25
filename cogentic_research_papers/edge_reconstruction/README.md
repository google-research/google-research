# Unconditional Edge Reconstruction of Simple Graphs of Linear Density

[This paper](paper.pdf) resolves Harary's (1964)
Edge Reconstruction Conjecture for all finite simple graphs of linear edge
density above an explicit constant threshold:

1. **Linear-density edge reconstruction (Theorem 1.1):** Every finite simple
   undirected graph $G_0$ on $n_0 \ge 4$ vertices with
   $m = |E(G_0)| \ge 1000\, n_0$ edges is edge-reconstructible (uniquely
   determined up to isomorphism by the multiset
   $\mathcal{D}_E(G_0) = \{\!\{[G_0 - e] : e \in E(G_0)\}\!\}$ of its
   unlabelled single-edge-deleted subgraphs).
2. **Minimum-degree corollary (Corollary 1.2):** Every finite simple undirected
   graph $G_0$ with minimum degree $\delta(G_0) \ge 2000$ is
   edge-reconstructible.

Previously, the strongest unconditional density bound valid for arbitrary simple
graphs was Müller's $m > n \log_2 n$ (1977), refining Lovász's
$m > \frac{1}{2}\binom{n}{2}$ (1972); other unconditional results require extra
structure, such as $m > n \log_2 \Delta(G) + O(n)$ or a bounded-degree spanning
tree (Caro, Krasikov, and Roditty 1985). A complete, unconditional, Mathlib-only
Lean 4 (`v4.33.0`) formalization of the entire proof has also been completed and
will be released in a companion submission.

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
