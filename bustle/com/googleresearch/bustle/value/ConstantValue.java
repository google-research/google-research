package com.googleresearch.bustle.value;

import java.util.Collections;

/**
 * A Value created from a constant.
 */
public final class ConstantValue extends Value {
  private final Object constant;

  public ConstantValue(Object constant, int numCopies) {
    super(Collections.nCopies(numCopies, constant));
    this.constant = constant;
  }

  @Override
  public String expression() {
    if (getType().equals(String.class)) {
      return "\"" + constant + "\"";
    } else if (getType().equals(Integer.class)) {
      return "" + constant;
    } else {
      throw new UnsupportedOperationException("Unhandled type in ConstantValue::expression");
    }
  }
}
