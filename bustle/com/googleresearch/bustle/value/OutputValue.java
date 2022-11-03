package com.googleresearch.bustle.value;

import java.util.List;

/**
 * A Value representing a desired output.
 */
public final class OutputValue extends Value {
  public OutputValue(List<Object> wrappedValues) {
    super(wrappedValues);
  }

  @Override
  public String expression() {
    throw new UnsupportedOperationException(
        "OutputValue intentionally does not implement expression()");
  }
}
