package com.googleresearch.bustle.value;

import java.util.List;

/**
 * A Value created from an input variable.
 */
public final class InputValue extends Value {
  private final String inputName;

  public InputValue(List<Object> wrappedValues, String inputName) {
    super(wrappedValues);
    this.inputName = inputName;
  }

  @Override
  public String expression() {
    return inputName;
  }
}
