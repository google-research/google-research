package com.googleresearch.bustle.serialization;

import com.google.common.collect.ImmutableList;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

/**
 * Custom Deserialization for WrappedValues lists. Needed because of int vs float issues: In
 * particular, if we have a List of Objects and they are originally Integers, Gson won't know to
 * save them as integers, and so will save them as floats. Happily, we do not support floats, and so
 * can just cast all float to int for now.
 */
public class WrappedValuesDeserializer implements JsonDeserializer<List<Object>> {

  @Override
  public List<Object> deserialize(
      JsonElement json, Type typeOfT, JsonDeserializationContext context) {

    JsonArray arr = json.getAsJsonArray();
    List<Object> finalResult;
    List<Object> resultsList = new ArrayList<>();
    for (JsonElement elt : arr) {
      if (elt != null && elt.isJsonPrimitive() && elt.getAsJsonPrimitive().isNumber()) {
        resultsList.add(elt.getAsInt());
      } else {
        resultsList.add(context.deserialize(elt, Object.class));
      }
    }
    finalResult = ImmutableList.copyOf(resultsList);
    return finalResult;
  }
}
