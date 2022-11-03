package com.googleresearch.bustle;

import com.googleresearch.bustle.propertysignatures.PropertySummary;
import com.googleresearch.bustle.value.Value;
import java.util.List;

/** Holds data for analyzing predictions of a model on the benchmarks */
public class PredictionDataItem {
  private final String expectedProgram;
  private final Value subExpression;
  private final String subExpressionString;
  private boolean isPositiveExample;
  private final double prediction;
  private final List<PropertySummary> exampleSignature;
  private final List<PropertySummary> valueSignature;

  public PredictionDataItem(
      String expectedProgram,
      Value subExpression,
      String subExpressionString,
      boolean isPositiveExample,
      double prediction,
      List<PropertySummary> exampleSignature,
      List<PropertySummary> valueSignature) {

    this.expectedProgram = expectedProgram;
    this.subExpression = subExpression;
    this.subExpressionString = subExpressionString;
    this.isPositiveExample = isPositiveExample;
    this.prediction = prediction;
    this.exampleSignature = exampleSignature;
    this.valueSignature = valueSignature;
  }

  public String getExpectedProgram() {
    return expectedProgram;
  }

  public Value getSubExpression() {
    return subExpression;
  }

  public String getSubExpressionString() {
    return subExpressionString;
  }

  public boolean getIsPositiveExample() {
    return isPositiveExample;
  }

  public void setIsPositiveExample(boolean isPositiveExample) {
    this.isPositiveExample = isPositiveExample;
  }

  public double getPrediction() {
    return prediction;
  }

  public List<PropertySummary> getExampleSignature() {
    return exampleSignature;
  }

  public List<PropertySummary> getValueSignature() {
    return valueSignature;
  }
}
