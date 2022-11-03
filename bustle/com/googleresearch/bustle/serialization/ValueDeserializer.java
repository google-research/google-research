package com.googleresearch.bustle.serialization;

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.googleresearch.bustle.value.ConstantValue;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.OperationValue;
import com.googleresearch.bustle.value.Value;
import java.lang.reflect.Type;

/**
 * Custom Deserialization for Values.
 *
 * <p>When processing training data for the Value Prediction model, we need to load a list of
 * arbitrary values from disk. Gson does not support this out of the box, so we use this function to
 * load the Json for a Value, check what sub-class of Value it actually corresponds to, and then
 * deserialize the Json as that sub-class.
 */
public class ValueDeserializer implements JsonDeserializer<Value> {

  @Override
  public Value deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) {

    JsonObject obj = json.getAsJsonObject();
    JsonElement jsonOperationName = obj.get("operation"); // non-null only if OperationValue
    JsonElement jsonInputName = obj.get("inputName"); // non-null only if InputValue
    JsonElement jsonConstant = obj.get("constant"); // non-null only if ConstantValue
    if (jsonOperationName != null) {
      return context.deserialize(json, OperationValue.class);
    } else if (jsonInputName != null) {
      return context.deserialize(json, InputValue.class);
    } else if (jsonConstant != null) {
      return context.deserialize(json, ConstantValue.class);
    } else {
      throw new IllegalArgumentException(
          "Can't deserialize Value sub-class other than OperationValue, InputValue, ConstantValue");
    }
  }
}
