package com.googleresearch.bustle.propertysignatures;

/**
 * Enum summarizing the result of a property function applied to inputs and outputs.
 */
public enum PropertySummary {
  ALL_TRUE(0), // The property returns true for all inputs.
  ALL_FALSE(1), // The property returns false for all inputs.
  MIXED(2), // The property returns true for some inputs and false for others.
  TYPE_MISMATCH(3), // The property could not be evaluated for the inputs due to type mismatch.
  EMPTY(4); // The property was not evaluated at all because there were no inputs.

  private final int intRepresentation; // A unique integer for representing the enum in a tensor.

  private PropertySummary(int intRepresentation) {
    this.intRepresentation = intRepresentation;
  }

  public int asInt() {
    return intRepresentation;
  }
}
