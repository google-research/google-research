package com.googleresearch.bustle;

/** Categories for benchmarks. A benchmark may have any number of tags. */
public enum BenchmarkTag {
  // The task involves constant strings extracted from the examples.
  CONSTANT,
  // The task involves a conditional.
  CONDITIONAL,
  // The task involves a regex.
  REGEX,
  // The task involves an array, either a literal {a, b, ...} or created from a function like SPLIT
  // or SEQUENCE, or uses an ArrayFormula.
  ARRAY,
  // The task involves formatting using TEXT(number, format).
  TEXT_FORMATTING,
  // This task may be excluded from experiments because the simplest known solution is still
  // unreasonably difficult.
  TOO_DIFFICULT,
  // The task is intended to be unsolvable.
  SHOULD_FAIL,
}
