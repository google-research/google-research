# Sketches

This repository contains a set of algorithms to summarize streaming data. The
stream consists of pairs (int key, float value). Each algorithm implements:
1. Update to update its internal state.
2. Estimate - returns for any key an estimate of the sum of values seen so far.
3  HeavyHitters - returns a list of the K keys with the largest total values.
4. Merge - combine two compatible sketches.

We implement several standard algorithms including CountMin, LossyCount and
Misra-Gries, and their variants.

## Example Usage

Build and run run_sketches.cc to run all programs.
bazel build -c opt run_sketches
bazel-bin/run_sketches
