package com.googleresearch.bustle.value;

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.googleresearch.bustle.Operation;
import java.util.List;

/**
 * A Value created by applying an Operation to arguments.
 */
public final class OperationValue extends Value {
  private final Operation operation;
  private final List<Value> arguments;

  public OperationValue(
      List<Object> wrappedValues, Operation operation, List<Value> arguments) {
    super(wrappedValues);
    this.operation = operation;
    this.arguments = ImmutableList.copyOf(arguments);
  }

  public Operation getOperation() {
    return operation;
  }

  public List<Value> getArguments() {
    return arguments;
  }

  @Override
  public String expression() {
    return operation.getName()
        + "("
        + arguments.stream().map(Value::expression).collect(joining(", "))
        + ")";
  }
}
