---
name: lemma_mathlib_tactics
description: >-
  Lean 4 and Mathlib formalization methodology, tactic execution, type coercions,
  and proof automation guide. Use when selecting proof tactics (omega, linarith,
  ring, grind, gcongr, aesop), handling casts and coercions, or structuring
  induction. For theorem/structure search and discovery, use lemma_mathlib_search.
depends_on:
  - lemma_mathlib_search
---

# Lean 4 & Mathlib Tactics & Automation Methodology

This skill provides tactical execution workflows, type coercion patterns, and algebraic solving techniques in Lean 4.

For discovering canonical structures, definitions, and lemma signatures across Mathlib, consult **`lemma_mathlib_search`**.

---

## 1. Tactical Execution & Domain Automation

When building formal proofs, select tactics based on the algebraic and logical goal class:

* **Linear Integer / Natural Arithmetic**: `omega` (handles Presburger arithmetic, linear equalities, and inequalities over `ℕ` and `ℤ`).
* **Linear Real / Rational Arithmetic**: `linarith` (for linear inequalities and equalities over ordered fields).
* **Polynomial Ring Identities**: `ring` / `ring_nf` (proves polynomial identities over commutative rings/semirings; separate sign facts first).
* **Nonlinear Real Inequalities**: `nlinarith` (linear arithmetic augmented with nonnegativity squares and product facts).
* **Rational Fractions & Denominators**: `field_simp` followed by `ring` or `linarith` (prove non-zero denominator side goals first).
* **Relational Monotonicity & Congruence**: `gcongr` / `congr` (automatically applies monotonicity along inequalities and congruences).
* **SMT & Local Structure Automation**: `grind` / `grind?` (Lean 4.27.0+ automated SMT-style solver for local context simplification).
* **General Proof Search**: `aesop` (best for inductive types, propositional logic, and structural constructors).
* **Decidable Finite Computation**: `decide` / `norm_num` (evaluates closed finite propositions and numerical expressions).

---

## 2. Specialized Tactical Guide

For comprehensive tactic tables, type-coercion patterns, and algebraic solving techniques, consult:
* **Mathlib Tactics Guide**: [`references/mathlib_tactics_guide.md`](references/mathlib_tactics_guide.md) (read on demand).

