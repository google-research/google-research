package com.googleresearch.bustle.serialization;

import com.google.gson.JsonElement;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import com.googleresearch.bustle.value.ConstantValue;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.OperationValue;
import com.googleresearch.bustle.value.Value;
import java.lang.reflect.Type;

/**
 * Custom Serialization for Values.
 *
 * <p>This class is needed to ensure that when we serialize a {@code List<Value>}, the on-disk
 * representation of each value is minimally polymorphic.
 */
public class ValueSerializer implements JsonSerializer<Value> {

  @Override
  public JsonElement serialize(Value src, Type typeOfT, JsonSerializationContext context) {

    if (src instanceof OperationValue) {
      return context.serialize(src, OperationValue.class);
    } else if (src instanceof InputValue) {
      return context.serialize(src, InputValue.class);
    } else if (src instanceof ConstantValue) {
      return context.serialize(src, ConstantValue.class);
    } else {
      throw new IllegalArgumentException(
          "Can't serialize Value sub-class other than OperationValue, InputValue, ConstantValue");
    }
  }
}
