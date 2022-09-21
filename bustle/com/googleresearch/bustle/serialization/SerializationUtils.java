package com.googleresearch.bustle.serialization;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import com.googleresearch.bustle.Operation;
import com.googleresearch.bustle.value.Value;
import java.lang.reflect.Type;
import java.util.List;

/** Utilities used for serialization/deserialization of programs */
public final class SerializationUtils {

  public static Gson constructCustomGsonBuilder() {
    Type wrappedValueType = new TypeToken<List<Object>>() {}.getType();
    return new GsonBuilder()
        .registerTypeHierarchyAdapter(Class.class, new ClassTypeAdapter())
        .registerTypeAdapter(wrappedValueType, new WrappedValuesDeserializer())
        .registerTypeAdapter(Operation.class, new OperationDeserializer())
        .registerTypeAdapter(Value.class, new ValueDeserializer())
        .registerTypeAdapter(Value.class, new ValueSerializer())
        .create();
  }

  private SerializationUtils() {}
}
