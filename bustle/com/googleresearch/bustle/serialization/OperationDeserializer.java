package com.googleresearch.bustle.serialization;

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.googleresearch.bustle.Operation;
import java.lang.reflect.Type;

/**
 * Custom Deserialization for Operations Since an operation contains a java.lang.reflect.Method
 * which we can't serialize, we skip serialization of the Method attribute of the Operation class by
 * marking it transient.
 *
 * <p>Then, during deserialization, we read the method name out of the JSON, look up the actual
 * method in a Map from names to Methods, and reconstruct an Operation using the method name we
 * looked up.
 */
public class OperationDeserializer implements JsonDeserializer<Operation> {

  @Override
  public Operation deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) {

    JsonObject obj = json.getAsJsonObject();
    JsonElement jsonName = obj.get("name");
    String opName = jsonName.getAsString();
    JsonElement jsonNumArgs = obj.get("numArgs");
    int numArgs = jsonNumArgs.getAsInt();
    Operation namedOp = Operation.lookupOperation(opName, numArgs);

    return new Operation(namedOp.getMethod());
  }
}
