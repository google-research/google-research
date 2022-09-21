package com.googleresearch.bustle;

import com.googleresearch.bustle.propertysignatures.ComputeSignature;
import com.googleresearch.bustle.propertysignatures.PropertySummary;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.Value;
import java.util.List;

/**
 * Class holding a 4-tuple (inputValues, subExpression, targetExpression, isPositiveExample) It also
 * holds 2 property signatures, one computed on the input and output and one computed using the
 * subExpression and the output. Finally, for premise selection, it holds a vector of booleans, one
 * for each operation (sorted first alphabetically by name and then by number of expressions)
 * indicating whether that operation was used in the subExpression.
 *
 * <p>This class will be used as data to train a model to predict whether an intermediate value can
 * be expanded into the desired final program.
 *
 * <p>It's not an AutoValue because that would make it hard for Gson to serialize it.
 */
public class TrainingDataItem {
  private final List<InputValue> inputValues;
  private final Value subExpression;
  private final Value targetExpression;
  private final boolean isPositiveExample;
  private final List<PropertySummary> exampleSignature;
  private final List<PropertySummary> valueSignature;
  private final List<Boolean> subExpressionOperationPresenceList;

  public TrainingDataItem(
      List<InputValue> inputValues,
      Value subExpression,
      Value targetExpression,
      boolean isPositiveExample) {
    this.inputValues = inputValues;
    this.subExpression = subExpression;
    this.targetExpression = targetExpression;
    this.isPositiveExample = isPositiveExample;
    this.exampleSignature =
        ComputeSignature.computeExampleSignature(inputValues, targetExpression);
    this.valueSignature =
        ComputeSignature.computeValueSignature(subExpression, targetExpression);
    this.subExpressionOperationPresenceList =
        Utils.getOperationPresenceListFromValue(subExpression);
  }

  public List<InputValue> getInputValues() {
    return inputValues;
  }

  public Value getSubExpression() {
    return subExpression;
  }

  public List<Boolean> getSubExpressionOperationPresenceList() {
    return subExpressionOperationPresenceList;
  }

  public Value getTargetExpression() {
    return targetExpression;
  }

  public boolean getIsPositiveExample() {
    return isPositiveExample;
  }

  public List<PropertySummary> getExampleSignature() {
    return exampleSignature;
  }

  public List<PropertySummary> getValueSignature() {
    return valueSignature;
  }
}
