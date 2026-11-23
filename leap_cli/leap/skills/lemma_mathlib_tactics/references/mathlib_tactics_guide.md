# General Lean 4 and Mathlib Proving Hints

## Purpose

This is a standalone reference for solving mathematical goals in Lean 4 with Mathlib. It applies whether the agent begins with a complete mathematical argument, a partial plan, or only the current proof state. It focuses on the gap between paper mathematics and formal proof: statements that are conceptually immediate may require careful choices of types, representations, coercions, induction predicates, theorem interfaces, and side-condition management.

The desired result is checked Lean code for the exact theorem in the actual project environment. Mathematical plausibility, a successful proof of a nearby statement, and a tactic that works only in a different import context are not sufficient.

The central discipline is:

> Read the exact proof state, make one semantically justified transformation, and compile immediately.

Do not reason about an imagined goal. Lean checks the elaborated types in the current environment, not the mathematical sentence the author intended.

Because Lean and Mathlib evolve, verify names, signatures, imports, and tactic availability in the actual project. Never rely on memory of an API when `#check`, documentation search, source search, or a scratch `example` can confirm it.

## Fast formalization protocol

1. Read the project’s Lean version, Mathlib revision, imports, namespaces, and local notation.
2. Copy the theorem into a minimal scratch `example` without changing its type.
3. Parse the target and every local hypothesis by type, not by variable name.
4. Choose representations that match existing APIs.
5. Search for the strongest relevant library theorem before reproving mathematics.
6. Prototype the highest-risk lemma or interface early.
7. Build helper lemmas with exact statements and compile them bottom-up.
8. Use structural tactics first, normalization second, domain automation third.
9. Inspect every generated subgoal, including side goals produced by rewriting or denominator clearing.
10. Replace exploration-only automation with a stable proof when practical; run the final checker with no placeholders.

## 1. The fundamental gap: informal mathematics versus Lean

| Informal phrase | What Lean may require |
| --- | --- |
| “Let `x` be a number.” | A precise type: `Nat`, `Int`, `Rat`, `Real`, or an abstract structure. |
| “Obviously the expressions are equal.” | Definitional equality (`rfl`), a rewrite theorem, normalization, or extensionality. |
| “Divide by `a`.” | A division operation in the type and a proof that `a ≠ 0`; inequalities also need a sign. |
| “Subtract `b` from both sides.” | Awareness that natural-number subtraction is truncated; perhaps work in `Int` or an ordered additive group. |
| “Take the maximum.” | A finite/nonempty set or compactness/attainment theorem, plus a representation of the maximizer. |
| “Choose such an element.” | An explicit witness, an existence theorem, or classical choice. |
| “The sets are equal.” | Usually `ext x; simp` or two membership implications. |
| “The functions are equal.” | `funext x` and a pointwise proof. |
| “Reorder the sum.” | A `Finset` equivalence/bijection or commutative big-operator theorem. |
| “Use induction.” | The correct motive, generalized variables, constructor cases, and suitable recursion principle. |
| “By symmetry, assume ...” | A symmetry map and a transport proof, or a `wlog`-style argument with explicit coverage. |
| “The result follows by computation.” | A decidable finite proposition, trusted evaluation mechanism allowed by the project, and a checked certificate. |
| “Apply theorem X.” | Exact namespace, argument order, implicit parameters, typeclass instances, and syntactic conclusion shape. |

The difficulty is often not proving a mathematical fact but making the fact appear in the exact typed form expected by the next theorem.

### When Lean should change the informal proof

Do not insist on a line-by-line translation when another certified route has a much better formal interface. It is often preferable to:

- invoke a more abstract library theorem and prove short adapter lemmas;
- replace a “without loss of generality” quotient/symmetry argument with explicit exhaustive cases;
- replace hand algebra by a polynomial identity plus separately proved sign facts;
- replace a chosen maximum by a finite induction that carries a witness;
- use pointwise extensionality instead of transporting whole structured objects;
- keep a calculation in one ambient number system rather than repeatedly moving between `Nat`, `Int`, and `Real`.

The correct formal proof should preserve the mathematical certificate, not necessarily the prose order. When the original route has a poor formal interface, restate the local proof plan honestly and use the Lean-friendly argument.

## 2. Start with the environment, not the proof idea

Before writing tactics, inspect:

- the `lean-toolchain` version and dependency revision;
- the actual imports available to the file;
- open namespaces and scoped notation;
- local variables and typeclass assumptions;
- existing project definitions that may differ from Mathlib’s;
- whether classical logic, noncomputability, or special axioms are permitted.

Useful inspection commands include:

```lean
#check theoremName
#check @theoremName
#print theoremName
#synth SomeTypeclass α
#eval someClosedComputation
```

`#check @name` exposes implicit parameters. Use editor hover and the proof state as primary evidence. If pretty-printing hides decisive coercions or implicit arguments, temporarily enable more explicit output locally.

Do not add a broad import merely to make an unknown tactic appear unless the project permits it. Imports affect compile time, namespace collisions, and available instances.

## 3. Read the proof state literally

A goal has a local context and a target:

```text
α : Type u
inst : LinearOrderedField α
x y : α
hx : 0 < x
⊢ x / y ≤ x
```

This state does **not** contain `0 < y`, `1 ≤ y`, or `y ≠ 0`. A human may infer an omitted intended assumption; Lean correctly refuses.

For every state, ask:

1. What is the head connective of the target?
2. Which hypotheses have exactly usable types?
3. Which expressions are definitionally equal, propositionally equal, or merely mathematically equivalent?
4. What metavariables, instances, and coercions might elaboration be trying to infer?
5. Will the proposed tactic reduce the semantic difficulty, or only change syntax?

After a tactic creates multiple goals, enumerate them. Never assume the visible first goal is the only remaining obligation.

## 4. Choose types and representations deliberately

### 4.1 `Nat`, `Int`, `Rat`, and `Real`

These are not interchangeable.

- `Nat` subtraction is truncated: `3 - 5 = 0`.
- Integer and field division have different semantics and theorem families.
- Casting an inequality may require a theorem showing the cast preserves order.
- A polynomial identity may be easiest in a semiring; an inequality may require a linear ordered ring or field.

If a proof relies on ordinary subtraction and cancellation, consider moving the local algebra into `Int` or another additive group, then transport back. Conversely, keep cardinality arguments in `Nat` when that matches finite APIs.

Use casts in controlled regions. Normalize them with appropriate cast lemmas or tactics available in the project; do not scatter coercions throughout every expression.

### 4.2 `Set`, `Finset`, `Fintype`, `List`, and indexed families

- `Set α` is a predicate `α → Prop`; it does not itself carry enumeration data.
- `Finset α` is finite data and generally needs decidable equality for elementwise computation.
- `[Fintype α]` provides an enumeration of an entire finite type.
- `List α` preserves order and multiplicity.
- `Multiset α` preserves multiplicity but not order.
- `Fin n → α` is often convenient for a fixed-length indexed family.

Choose based on downstream operations. A combinatorial proof involving cardinality, filtering, and sums often wants `Finset`; a subsequence proof needs order-preserving indices, not merely a set of values.

### 4.3 Subtypes and bundled structures

An element `x : {a // P a}` contains both a value `x.1` and a proof `x.2`. Coercions often display the value automatically, but equalities between subtypes may need extensionality. Bundled homomorphisms, subgroups, submodules, continuous maps, and equivalences carry laws and have specialized APIs; use these instead of rebuilding their properties pointwise.

### 4.4 Propositions versus booleans

`P : Prop` and `b : Bool` are different. Decidable propositions can be computed, but theorem interfaces may require bridging lemmas. Avoid switching representations unless computation provides a real benefit.

## 5. Equality: determine which kind you have

### Definitional equality

Two terms are definitionally equal when Lean can reduce them to the same expression. Then `rfl`, `change`, or `show` may work.

### Propositional equality

If terms are mathematically equal but not definitionally equal, supply a proof and use it with `rw`, `simp`, `calc`, congruence, or substitution.

Do not expect `rfl` to prove commutativity, associativity, extensional equality, quotient equality, or a theorem about a recursively defined function unless reduction actually exposes identical terms.

### Extensional equality

Use the object’s extensionality principle:

```lean
-- functions
funext x

-- sets or structures with registered extensionality lemmas
ext x
```

After `ext`, inspect the new goal. It may be membership equivalence, coordinate equality, or pointwise equality. `ext` is a reducer, not a guaranteed closer.

## 6. Structure the proof before using automation

Match the target’s logical shape.

| Target | Typical structural move |
| --- | --- |
| `∀ x, P x` | `intro x` |
| `P → Q` | `intro h` |
| `P ∧ Q` | `constructor` |
| `P ↔ Q` | `constructor` |
| `∃ x, P x` | `refine ⟨witness, ?_⟩` or `use witness` |
| equality chain | `calc ...` |
| structured object equality | `ext` / `funext` |
| disjunction/cases hypothesis | `rcases` / `cases` |
| conditional proposition | `by_cases h : P` or split the conditional |

Prefer `refine` when constructing a theorem application or object with known holes:

```lean
refine someTheorem ?premise₁ ?premise₂
```

The resulting goals document the interface. This is usually more informative than repeatedly applying tactics until something matches.

Use `have` for a forward fact and `suffices` for a backward bridge. Name facts by mathematical role (`h_nonzero`, `h_bound`, `h_inj`), not by chronology.

## 7. Rewriting and simplification without losing control

### `rw`

Use `rw [h]` when you know the equality and direction. Use `rw [← h]` for the reverse direction. Rewriting is syntactic up to elaboration and definitional equality; an equality can be mathematically relevant but fail to match the target’s current shape.

Repairs for a failed rewrite:

1. inspect `#check h` and the target;
2. expose or fold the relevant definition;
3. use `change` if the target is definitionally equal to the desired shape;
4. normalize casts or algebra first;
5. use a `calc` block to state the intermediate expression explicitly;
6. target a specific occurrence with occurrence control or `conv` when necessary.

### `simp`

`simp` uses oriented simplification lemmas and supplied facts. It is excellent for canonical cleanup, constructor reduction, memberships, identities, and closing routine side goals.

Guidelines:

- Use `simp [definition, h]` when a small unfolding is intended.
- Use `simp only [...]` when proof stability and auditability matter.
- Use `simpa [facts] using h` to bridge a small representational mismatch.
- Avoid dumping a large context into `simp` without understanding which rewrite is decisive.
- Do not add an unproven or poorly oriented global `[simp]` lemma to solve one local proof.
- If `simp` transforms the goal but does not close it, read the new goal rather than repeating `simp`.

Over-unfolding can destroy the abstract shape required by a library theorem. Unfold at the boundary of a custom definition, then return to standard interfaces.

## 8. High-value tactics by goal class

Tactic availability depends on imports and project versions. Confirm it locally.

| Goal class | Useful tools | Limits and preparation |
| --- | --- | --- |
| exact local/library fact | `exact`, `assumption`, `simpa`, `apply`, `refine` | The conclusion must unify in the correct direction. |
| propositional structure | `intro`, `constructor`, `rcases`, `rintro`, `by_cases`, `by_contra`, `tauto`/`aesop` when available | First expose the intended logic; watch constructive/classical requirements. |
| definitional cleanup | `rfl`, `dsimp`, `unfold`, `change`, `simp` | Definitional equality is narrower than mathematical equality. |
| equality transformation | `rw`, `calc`, `congr`, `ext`, `funext`, `conv` | Align expression shape and direction. |
| concrete numerals | `norm_num`, `decide` for suitable decidable closed goals | Does not invent missing mathematical structure. |
| polynomial identity | `ring`, `ring_nf`, sometimes `noncomm_ring` | Identities, not order reasoning or division with unknown denominators. |
| linear ordered algebra | `linarith` | Needs linear expressions over a supported ordered semiring/ring/field context. |
| polynomial ordered algebra | `nlinarith` | Needs polynomial structure and usually explicit sign/nonnegativity facts. |
| Presburger arithmetic | `omega` when available | Suited to linear arithmetic over naturals/integers, not nonlinear products. |
| rational expressions | `field_simp`, followed by `ring`/`linarith` | Generates or needs nonzero denominator obligations. |
| positivity/sign goals | `positivity` or explicit order lemmas | Cannot prove positivity absent sufficient assumptions. |
| finite explicit cases | `fin_cases`, `interval_cases`, `decide`/computation where suitable | Ensure the space is genuinely finite and the proof method is accepted. |
| continuity/measurability | domain tactics such as `continuity` or `measurability` when imported | Regularity of components and correct function shape are required. |
| relational & congruence goals | `gcongr`, `congr` | Applies monotonicity and congruence for relations (`≤`, `<`, `=`). |
| structured local automation | `grind`/`grind?` when available | Useful after relevant facts and definitions are exposed; verify the generated proof and project support. |
| broad proof search | `aesop`, `solve_by_elim`, suggestion tactics | Control depth and context; extract a stable proof when possible. |

Automation is strongest after the mathematical structure has been exposed and expressions normalized into its supported fragment.

### Common automation mismatches

- `linarith` cannot use a nonlinear product merely because the final conclusion is linear.
- `nlinarith` may need explicit facts such as `sq_nonneg x` or denominator signs.
- `ring` proves identities; it does not prove `x² ≥ 0`.
- `field_simp` is not permission to ignore zero denominators.
- `norm_num` handles concrete normalization, not symbolic theorem discovery.
- `grind` still needs the relevant facts and definitions in a usable form; it does not discover a missing representation or induction invariant.
- `aesop` may search the wrong space when a representation change is required.

## 9. Theorem search is part of proving

Do not hallucinate theorem names. Search by both concept and type shape.

### Search workflow

1. Identify the principal definitions in the goal.
2. Guess the namespace and inspect documentation/source search.
3. Use editor completion and `#check` candidate names.
4. Use suggestion tools such as `exact?`, `apply?`, or `simp?` when available.
5. Search by expected type shape when name search fails.
6. Inspect neighboring declarations in the source; theorem families often occur together.
7. Specialize the candidate in a scratch `example` before integrating it.

Search for the theorem in the form Mathlib prefers, not only the paper form. For instance, a target `a - b = c` may have a library theorem stated as `a = c + b`, with assumptions controlling subtraction.

When a suggestion tactic succeeds, keep the explicit theorem or simplification set it proposes when this improves stability and auditability.

## 10. Induction: the most common structural trap

Before `induction`, write the intended motive in mathematics.

### Weak induction hypothesis

If the step changes another variable, generalize it:

```lean
induction xs generalizing ys with
| nil => ...
| cons x xs ih => ...
```

If the recursive function follows a nontrivial recursion pattern, consider its functional induction principle rather than induction on an arbitrary visible argument. Recent Lean versions provide functional induction support, but verify the syntax and generated principle in the project.

### Destructive case split

`cases e` may replace the expression and make the original equation hard to recover. Use an equation-producing form when the relationship matters:

```lean
cases h : e with
| ... => ...
```

### Dependent hypotheses

When hypotheses depend on the induction variable, reverting/generalizing them changes the induction hypothesis. Inspect the generated state carefully; do not assume paper induction translates verbatim.

### Well-founded recursion

For descent, recursive definitions, or induction on a custom measure, isolate:

- the measure;
- proof that each recursive call decreases;
- the generalized predicate;
- equations describing the recursive function.

Often it is cheaper to prove the theorem by well-founded induction without first defining a complicated recursive program.

## 11. Side conditions Lean will not forgive

### Nonzero denominators

Prove and name them before `field_simp` or cancellation:

```lean
have hy : y ≠ 0 := ...
```

### Sign conditions

Multiplying inequalities needs positive/nonnegative or negative information. Do not expect automation to infer it through opaque definitions.

### Nonempty finite objects

Maxima, minima, and witness extraction may need explicit membership and nonemptiness. A theorem about `Finset.max'` differs from one about a default-valued maximum.

### Decidable equality and classical logic

Finset operations and equality tests may require `[DecidableEq α]`. Prefer local classical reasoning when acceptable:

```lean
classical
```

This supplies classical decidability and choice locally, but it does not prove mathematical existence conditions.

### Typeclass instances

If Lean cannot synthesize an algebraic/order/topological instance, first check whether the required assumptions exist and whether the intended type was inferred. Adding an instance blindly can create ambiguity or encode the wrong mathematics.

## 12. Common failure loops and how to escape

### Repeating tactics without reading state changes

**Loop:** try `simp`, `linarith`, `aesop`, and rewriting in changing orders.

**Escape:** classify the remaining target; isolate the missing mathematical fact as a `have` or helper lemma.

### Theorem-name guessing

**Loop:** generate plausible variants of a lemma name.

**Escape:** search documentation/source and verify with `#check`.

### Coercion whack-a-mole

**Loop:** add casts locally until each error moves elsewhere.

**Escape:** select one ambient type for a coherent calculation, prove a cast bridge once, normalize at the boundary.

### Unfolding everything

**Loop:** unfold until the target becomes enormous and no theorem matches.

**Escape:** return to a characterization theorem or prove a single adapter lemma at the abstraction boundary.

### Induction too early

**Loop:** every branch has a useless induction hypothesis.

**Escape:** revert, restate a stronger simultaneous theorem, or redesign the induction state before continuing.

### Giant automated close

**Loop:** increase search depth or timeout on an unstructured goal.

**Escape:** expose constructors, witnesses, cases, and core arithmetic; use automation only on small leaves.

### Proving the wrong nearby statement

**Loop:** much code establishes a coercion-shifted, reversed, or weakened fact that does not close the theorem.

**Escape:** use `show` or an explicit `have : ExactTarget := ...` to pin the intended interface.

## 13. Error-directed debugging

### Parse error

Check syntax, delimiters, indentation, and version-specific tactic grammar. Do not modify mathematics yet.

### Unknown identifier

Check spelling, namespace, import, version, and whether the declaration is local. Search rather than invent.

### Application type mismatch

Inspect the function/theorem type with `#check @name`. Compare each supplied argument to its expected type, including coerced types.

### Failed to synthesize instance

Identify the exact missing class. Check the inferred carrier type and existing assumptions. Decide whether an instance should be derivable, explicitly passed, or mathematically added.

### Tactic made no progress

The tactic’s supported fragment does not match the current shape, or required facts are absent. Normalize, expose definitions selectively, or prove a missing premise.

### Unsolved goals

List every goal and identify which tactic created it. Side goals often encode the nonzero, positivity, membership, or well-foundedness fact that the informal proof omitted.

### Timeout or heartbeat exhaustion

Reduce the context, avoid huge simplifier sets, split the theorem, provide key lemmas, or replace broad search with a targeted decision procedure. Longer runtime is rarely the first repair.

## 14. Formalization patterns with high leverage

### Adapter lemma

```lean
have h_standard : StandardPredicate x := by
  -- translate custom hypotheses once
  ...
exact standard_theorem h_standard
```

### Calculation chain

```lean
calc
  lhs = middle₁ := by ...
  _ ≤ middle₂ := by ...
  _ = rhs := by ...
```

This fixes orientations and localizes failures.

### Witness plus certificate

```lean
refine ⟨candidate, ?_⟩
-- prove exactly the candidate's required property
```

### Extract structured hypotheses

```lean
rcases h_exists with ⟨x, hx₁, hx₂⟩
rintro ⟨hP, hQ⟩
```

### Local representation boundary

```lean
let S : Finset α := ...
have hS : ... := by
  -- all Finset-specific work here
...
```

### Separate identity from order

First prove a polynomial identity with `ring`; then combine it with sign facts using order lemmas or arithmetic automation. This mirrors the mathematical certificate and is usually more robust than asking one tactic to discover both.

## 15. Domain-specific Lean advice

### Finite combinatorics

- Decide early whether vertices/indices are a `Finset`, a `Fintype`, or `Fin n`.
- Use established cardinality lemmas and equivalences rather than manual list enumeration.
- Keep membership and distinctness proofs attached to the representation.
- For pigeonhole, choose domain and codomain so the library’s injective/surjective/cardinality theorem applies directly.
- Avoid turning ordered subsequences into unordered sets.

### Algebra and number theory

- State the weakest useful algebraic typeclass, but do not over-generalize a one-off theorem if it multiplies instance problems.
- Distinguish explicit divisibility witnesses, modular equivalence, ideal membership, and equality in quotient rings.
- Normalize casts at boundaries and use domain-appropriate arithmetic automation.
- Separate zero cases before cancellation.

### Analysis

- Prefer library notions of continuity, convergence, compactness, integrability, and measurability.
- Search for closure/composition theorems before unfolding epsilon/filter definitions.
- Explicitly identify filters and modes of convergence when goals appear ambiguous.
- Expect side conditions for nonempty compact sets, eventual properties, measurability, and integrability.
- Avoid manual epsilon proofs unless the required theorem is missing or the direct definition is genuinely simpler.

### Linear algebra and abstract structures

- Use bundled maps and extensionality lemmas.
- Expect coercions from morphisms to functions and from subobjects to carrier types.
- Prefer basis-free universal properties when APIs support them; coordinates introduce finite sums and index transport.
- Track commutativity and scalar-tower assumptions explicitly.

## 16. Proof robustness and maintainability

A checked proof can still be fragile. Prefer proofs that:

- name central intermediate facts;
- use explicit theorem names at important interfaces;
- constrain `simp` when a large global simp set is unnecessary;
- avoid reliance on autogenerated hypothesis names;
- avoid unexplained long tactic chains;
- keep classical reasoning local;
- compile with the project’s minimal intended imports;
- use public APIs rather than implementation details;
- preserve the mathematical decomposition in the code structure.

Do not prematurely golf a proof. First obtain a clear checked proof, then shorten only where readability and stability improve.

## 17. Verification and acceptance

Before declaring success:

- compile the exact target theorem in the actual project;
- ensure no `sorry`, `admit`, unsound axiom, or prohibited placeholder remains;
- inspect warnings relevant to deprecations, unused variables, or declarations;
- confirm all imports are permitted;
- check that the theorem statement itself was not weakened or altered;
- record any classical assumptions or noncomputable definitions introduced;
- if using computation, verify that the project accepts the mechanism and that the proposition-to-computation equivalence is checked;
- rerun the relevant file or project test command from a clean proof state.

A proof is complete only when the compiler checks the intended theorem, not when a similar scratch example succeeds.

## 18. Recommended proof record

```markdown
### Environment
- Lean/Mathlib version:
- Imports:
- Relevant namespaces/notations:

### Exact goal state
[Context and target copied from Lean.]

### Representation decisions
[Types chosen and rejected alternatives.]

### Library search
[Verified declarations and signatures.]

### Proof plan
[Structural steps and helper lemmas.]

### Implementation log
1. [Tactic/code change]
2. [Resulting goal or compiler result]
3. [Next mathematically justified step]

### Verification
- Exact theorem compiles:
- No placeholders:
- Edge cases:
- Imports and assumptions:
```

## Final checklist

- [ ] I am working in the correct Lean/Mathlib environment.
- [ ] I have not changed the theorem statement.
- [ ] I read exact types rather than inferring meaning from names.
- [ ] Representations match downstream library APIs.
- [ ] Every theorem name and signature has been verified.
- [ ] Casts are localized and domain semantics are correct.
- [ ] Nonzero, sign, nonempty, and decidability conditions are explicit.
- [ ] Induction has the correct generalized motive.
- [ ] Rewriting direction and target shape are controlled.
- [ ] Automation is applied only after structural preparation.
- [ ] All generated subgoals have been inspected.
- [ ] No placeholder or prohibited axiom remains.
- [ ] The exact file compiles successfully.

## Authoritative references

- [Lean Language Reference: Tactic Proofs](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/)
- [Lean Language Reference: Tactic Reference](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/Tactic-Reference/)
- [Lean Language Reference: Reading Proof States](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/Reading-Proof-States/)
- [Mathematics in Lean](https://leanprover-community.github.io/mathematics_in_lean/)
- [Mathlib documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [Searching for Theorems in Mathlib](https://leanprover-community.github.io/blog/posts/searching-for-theorems-in-mathlib/)
