package com.googleresearch.bustle.serialization;

import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;

/** Tells gson how to serialize and deserialize Class<?> */
public class ClassTypeAdapter extends TypeAdapter<Class<?>> {
  @Override
  public Class<?> read(JsonReader in) throws IOException {
    if (in.peek() == JsonToken.NULL) {
      in.nextNull();
      return null;
    } else {
      String className = in.nextString();
      try {
        return Class.forName(className);
      } catch (ClassNotFoundException e) {
        throw new JsonParseException("class " + className + " not found", e);
      }
    }
  }

  @Override
  public void write(JsonWriter out, Class<?> value) throws IOException {
    if (value == null) {
      out.nullValue();
    } else {
      out.value(value.getName());
    }
  }
}
