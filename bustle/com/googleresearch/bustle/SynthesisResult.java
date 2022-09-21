package com.googleresearch.bustle;

import com.google.auto.value.AutoValue;

/** The result of running the synthesizer once. */
@AutoValue
public abstract class SynthesisResult {

  public static SynthesisResult createSuccess(
      String benchmarkName,
      int numExpressionsTried,
      int numUniqueValues,
      double elapsedTime,
      String solution,
      int solutionWeight) {
    return new AutoValue_SynthesisResult(
        benchmarkName,
        numExpressionsTried,
        numUniqueValues,
        elapsedTime,
        true,
        solution,
        solutionWeight);
  }

  public static SynthesisResult createFailure(
      String benchmarkName, int numExpressionsTried, int numUniqueValues, double elapsedTime) {
    return new AutoValue_SynthesisResult(
        benchmarkName, numExpressionsTried, numUniqueValues, elapsedTime, false, "", -1);
  }

  public abstract String getBenchmarkName();
  public abstract int getNumExpressionsTried();
  public abstract int getNumUniqueValues();
  public abstract double getElapsedTime();
  public abstract boolean getSuccess();
  public abstract String getSolution();
  public abstract int getSolutionWeight();
}
