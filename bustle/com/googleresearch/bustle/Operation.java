package com.googleresearch.bustle;

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.toList;

import com.google.common.base.Ascii;
import com.googleresearch.bustle.exception.SynthesisError;
import com.googleresearch.bustle.value.OperationValue;
import com.googleresearch.bustle.value.Value;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** An operation supported by the Sheets synthesizer. */
public final class Operation {
  // ``transient'' keyword prevents Gson from trying to serialize this field
  private final transient Method method;
  private final String name;
  private final int numArgs;
  private final List<Class<?>> argTypes;
  private final Class<?> returnType;

  public Operation(Method method) {
    this.method = method;
    name = Ascii.toUpperCase(method.getName());
    argTypes = Arrays.asList(method.getParameterTypes());
    returnType = method.getReturnType();
    numArgs = argTypes.size();
  }

  // We need this getter in the custom Operation deserializer in OperationDeserializer.java
  public Method getMethod() {
    return method;
  }

  public String getName() {
    return name;
  }

  public int getNumArgs() {
    return numArgs;
  }

  public List<Class<?>> getArgTypes() {
    return argTypes;
  }

  public Class<?> getReturnType() {
    return returnType;
  }

  public Value apply(List<Value> args) {
    if (args.size() != numArgs) {
      throw new SynthesisError(
          String.format(
              "Cannot apply operation %s on %d arguments (expected %d arguments)",
              name, args.size(), numArgs));
    }
    try {
      int numExamples = args.get(0).getNumWrappedValues();
      List<Object> resultObjects = new ArrayList<>();
      for (int idx = 0; idx < numExamples; idx++) {
        Object[] theseArgs = new Object[numArgs];
        for (int i = 0; i < numArgs; i++) {
          theseArgs[i] = args.get(i).getWrappedValue(idx);
        }
        Object invocationResult = method.invoke(null, theseArgs);
        if (invocationResult == null) {
          return null;
        }
        resultObjects.add(invocationResult);
      }
      return new OperationValue(resultObjects, this, args);
    } catch (InvocationTargetException e) {
      return null;
    } catch (IllegalAccessException e) {
      throw new SynthesisError("IllegalAccessException caught: " + e);
    }
  }

  public static List<Operation> getOperations() {
    return Arrays.stream(SheetsFunctions.class.getDeclaredMethods())
        .sorted(comparing(Method::getName).thenComparing(Method::getParameterCount))
        .map(Operation::new)
        .collect(toList());
  }

  public static Operation lookupOperation(String name, int numArgs) {
    for (Operation o : Operation.getOperations()) {
      if (o.getName().equals(name) && o.getNumArgs() == numArgs) {
        return o;
      }
    }
    throw new IllegalArgumentException(
        "Operation " + name + " with " + numArgs + " arguments does not exist.");
  }
}
