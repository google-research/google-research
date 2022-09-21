package com.googleresearch.bustle;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.googleresearch.bustle.value.OperationValue;
import java.util.List;

/**
 * A simple class to represent one training data instance. Note that the OperationValue contains the
 * results of a code expression run on all examples for the same problem. The OperationValue is one
 * value encountered during an enumerative search starting from the inputs in the examples. The
 * outputs of the examples match the OperationValue.
 */
@AutoValue
public abstract class DataItem {

  public static DataItem create(List<Example> examples, OperationValue value) {
    return new AutoValue_DataItem(ImmutableList.copyOf(examples), value);
  }

  public abstract ImmutableList<Example> getExamples();
  public abstract OperationValue getValue();
}
