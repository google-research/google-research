package com.googleresearch.bustle;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import java.util.List;

/** A class to define an input-output example. */
@AutoValue
public abstract class Example {

  public static Example create(List<String> inputs, String output) {
    return new AutoValue_Example(ImmutableList.copyOf(inputs), output);
  }

  public abstract ImmutableList<String> inputs();

  public abstract String output();

  /** Returns the string representation for the input-output example. */
  @Override
  public final String toString() {
    return String.join("|", inputs()) + " --> " + output();
  }
}
