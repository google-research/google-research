package com.googleresearch.bustle.value;

import com.google.common.collect.ImmutableList;
import java.util.List;

/**
 * An intermediate value in Sheets synthesis.
 */
public abstract class Value {
  private final List<Object> wrappedValues;
  private final Class<?> type;
  private final int numWrappedValues;

  public Value(List<Object> wrappedValues) {
    this.wrappedValues = ImmutableList.copyOf(wrappedValues);
    numWrappedValues = wrappedValues.size();
    this.type = wrappedValues.get(0).getClass();
  }

  public List<Object> getWrappedValues() {
    return wrappedValues;
  }

  public Object getWrappedValue(int index) {
    return wrappedValues.get(index);
  }

  public Class<?> getType() {
    return type;
  }

  public int getNumWrappedValues() {
    return numWrappedValues;
  }

  @Override
  public int hashCode() {
    return wrappedValues.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof Value)) {
      return false;
    }
    return wrappedValues.equals(((Value) obj).wrappedValues);
  }

  public abstract String expression();

  @Override
  public String toString() {
    return this.getClass().getSimpleName() + "(" + expression() + ")" + wrappedValues;
  }
}
