# Lean 4 & Mathlib Search & Representation Guide

## 1. Core Representation Principle: Canonical Structures > Raw Data

Mathlib provides hundreds of algebraic, topological, and order simplification lemmas for its canonical structures. When you model a problem using raw tuples or arrays instead of Mathlib's native types, you lose access to these lemmas and are forced into manual proofs.

### Canonical Structure Discovery Guide:

| Mathematical Domain | Problem Concept | Canonical Mathlib Type | Discovery Search Query |
| :--- | :--- | :--- | :--- |
| **Combinatorics** | $k$-element subsets | `Finset.powersetCard k s` or `Sym k α` | `query="powersetCard cardinality"` |
| | Set Partitions | `Finpartition s`, `Set.PairwiseDisjoint` | `query="Finpartition ofExistsUnique"` |
| | Symmetries / Actions | `Equiv.Perm α`, `MulAction` | `query="cardinality of Equiv.Perm"` |
| | Simple Graphs | `SimpleGraph V`, `SimpleGraph.Coloring` | `query="SimpleGraph neighborSet"` |
| **Analysis** | Convexity & Jensen | `Convex 𝕜 s`, `ConvexOn 𝕜 s f` | `query="ConvexOn sum Jensen"` |
| | Limits & Filters | `Filter.Tendsto`, `Filter.atTop` | `query="Filter.tendsto_atTop_atTop"` |
| | Metric Spaces | `MetricSpace α`, `Metric.ball` | `query="Metric ball isOpen"` |
| | Sumsets | `Pointwise` (`A + B`, `a • B`) | `query="cardinality of pointwise sumset"` |
| **Number Theory** | Base Expansions | `Nat.digits b n`, `Nat.ofDigits b L` | `query="Nat.ofDigits digits"` |
| | Divisibility | `Nat.factorization`, `Nat.Coprime` | `query="Nat.factorization dvd"` |
| | Series Divergence | `Summable f`, `¬Summable (fun n => 1/n)` | `query="Summable one div Nat"` |
| **Algebra** | Morphisms | `LinearMap`, `RingHom`, `MonoidHom` | `query="LinearEquiv map_add"` |
| | Submodules / Ideals | `Submodule R M`, `Ideal R` | `query="QuotientGroup Ideal.Quotient"` |

---

## 2. Mathlib Naming Grammar (Search Formulator)

Mathlib naming follows strict morphological rules. Use these suffixes to construct targeted search queries:

* `_iff`: Bidirectional equivalence ($\iff$).
* `_eq`: Exact equality ($=$).
* `_le` / `_lt`: Inequality relations ($\le, <$).
* `_of_`: Implication from a hypothesis (`A_of_B_of_C`).
* `_comm`: Commutativity (`Finset.sum_comm`, `add_comm`, `mul_comm`).
* `_assoc`: Associativity (`add_assoc`, `mul_assoc`).
* `_fiberwise`: Grouping summation by fiber partition (`Finset.sum_fiberwise_of_maps_to`).
* `_card` / `_ncard`: Cardinality of finite sets (`Finset.card_image_of_injective`).
* `_inj` / `_injective`: Injectivity property.
* `_mono` / `_strictMono`: Monotonicity under order.

---

## 3. Interactive In-Scratch Probing Patterns

Inside `nodes/<node_id>/scratch/attempt_1.lean`:

```lean
-- 1. exact? : Let Lean search Mathlib for the exact closing lemma
example (A B : Finset ℕ) : (A ∩ B).card ≤ A.card := by
  exact?

-- 2. apply? : Search for theorems that match goal conclusion
example (A B : Finset ℕ) : (A ∩ B).card ≤ A.card := by
  apply?

-- 3. #check @name : Inspect full signature with implicit parameters
#check @Finset.sum_fiberwise_of_maps_to

-- 4. #synth : Verify typeclass instance availability
#synth Fintype (Equiv.Perm (Fin 5))
```
