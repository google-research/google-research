package com.googleresearch.bustle.propertysignatures;

import com.google.common.base.Ascii;
import com.googleresearch.bustle.exception.SynthesisError;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.Value;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Computes a property signature for (inputs, output, intermediate value). */
public final class ComputeSignature {

  public static final int MAX_INPUTS = 3;

  public static final int NUM_STRING_PROPERTIES = 14;
  public static final int NUM_INT_PROPERTIES = 7;
  public static final int NUM_BOOL_PROPERTIES = 1;
  public static final int NUM_STRING_COMPARISON_PROPERTIES = 17;
  public static final int NUM_INT_COMPARISON_PROPERTIES = 7;
  public static final int NUM_SINGLE_VALUE_PROPERTIES =
      NUM_STRING_PROPERTIES + NUM_INT_PROPERTIES + NUM_BOOL_PROPERTIES;
  public static final int NUM_COMPARISON_PROPERTIES =
      NUM_STRING_COMPARISON_PROPERTIES + NUM_INT_COMPARISON_PROPERTIES;

  public static List<PropertySummary> computeExampleSignature(
      List<InputValue> inputs, Value output) {
    // Inputs and output must all be strings.
    if (!output.getType().equals(String.class)) {
      throw new SynthesisError("Property signature computation assumes the output is a String.");
    }
    for (InputValue input : inputs) {
      if (!input.getType().equals(String.class)) {
        throw new SynthesisError("Property signature computation assumes every input is a String.");
      }
    }

    List<InputValue> paddedInputs = new ArrayList<>(inputs);
    while (paddedInputs.size() < MAX_INPUTS) {
      paddedInputs.add(null);
    }

    int size =
        (MAX_INPUTS + 1) * (NUM_STRING_PROPERTIES + NUM_INT_PROPERTIES + NUM_BOOL_PROPERTIES)
            + MAX_INPUTS * (NUM_STRING_COMPARISON_PROPERTIES + NUM_INT_COMPARISON_PROPERTIES);
    List<PropertySummary> signature = new ArrayList<>(size);

    // Signature will look like:
    // singleValue(in_1) | comparison(in_1, out) | singleValue(in_2) | comparison(in_2, out) ...
    // | singlevalue(out)
    for (Value input : paddedInputs) {
      processSingleValue(input, signature);
      processComparison(input, output, signature);
    }
    processSingleValue(output, signature);

    return signature;
  }

  public static List<PropertySummary> computeValueSignature(Value intermediate, Value output) {
    int size =
        NUM_STRING_PROPERTIES
            + NUM_INT_PROPERTIES
            + NUM_STRING_COMPARISON_PROPERTIES
            + NUM_INT_COMPARISON_PROPERTIES;
    List<PropertySummary> signature = new ArrayList<>(size);
    // Signature will look like:
    // singleValue(intermediate) | comparison(intermediate, out)
    processSingleValue(intermediate, signature);
    processComparison(intermediate, output, signature);
    return signature;
  }

  private static void processSingleValue(Value value, List<PropertySummary> signature) {
    Class<?> type = value == null ? void.class : value.getType();

    if (type.equals(String.class)) {
      List<boolean[]> propertyResults = new ArrayList<>(); // num examples x num properties
      for (Object strObject : value.getWrappedValues()) {
        String str = (String) strObject;
        String lower = Ascii.toLowerCase(str);
        String upper = Ascii.toUpperCase(str);

        boolean[] properties =
            new boolean[] {
              str.isEmpty(), // is empty?
              str.length() == 1, // is single char?
              str.length() <= 5, // is short string?
              str.equals(lower), // is lowercase?
              str.equals(upper), // is uppercase?
              str.contains(" "), // contains space?
              str.contains(","), // contains comma?
              str.contains("."), // contains period?
              str.contains("-"), // contains dash?
              str.contains("/"), // contains slash?
              str.matches(".*\\d.*"), // contains digits?
              str.matches("\\d+"), // only digits?
              str.matches(".*[a-zA-Z].*"), // contains letters?
              str.matches("[a-zA-Z]+"), // only letters?
            };
        if (properties.length != NUM_STRING_PROPERTIES) {
          throw new SynthesisError(
              "NUM_STRING_PROPERTIES is "
                  + NUM_STRING_PROPERTIES
                  + " but should be "
                  + properties.length);
        }
        propertyResults.add(properties);
      }
      reducePropertyBooleans(propertyResults, signature);
    } else {
      signature.addAll(Collections.nCopies(NUM_STRING_PROPERTIES, PropertySummary.TYPE_MISMATCH));
    }

    if (type.equals(Integer.class)) {
      List<boolean[]> propertyResults = new ArrayList<>(); // num examples x num properties
      for (Object intObject : value.getWrappedValues()) {
        int integer = (int) intObject;
        boolean[] properties =
            new boolean[] {
              integer == 0, // is zero?
              integer == 1, // is one?
              integer == 2, // is two?
              integer < 0, // is negative?
              0 < integer && integer <= 3, // is small integer?
              3 < integer && integer <= 9, // is medium integer?
              9 < integer, // is large integer?
            };
        if (properties.length != NUM_INT_PROPERTIES) {
          throw new SynthesisError(
              "NUM_INT_PROPERTIES is "
                  + NUM_INT_PROPERTIES
                  + " but should be "
                  + properties.length);
        }
        propertyResults.add(properties);
      }
      reducePropertyBooleans(propertyResults, signature);
    } else {
      signature.addAll(Collections.nCopies(NUM_INT_PROPERTIES, PropertySummary.TYPE_MISMATCH));
    }

    if (type.equals(Boolean.class)) {
      List<boolean[]> propertyResults = new ArrayList<>(); // num examples x num properties
      for (Object boolObject : value.getWrappedValues()) {
        boolean bool = (boolean) boolObject;
        boolean[] properties =
            new boolean[] {
              bool, // is the boolean true?
            };
        if (properties.length != NUM_BOOL_PROPERTIES) {
          throw new SynthesisError(
              "NUM_BOOL_PROPERTIES is "
                  + NUM_BOOL_PROPERTIES
                  + " but should be "
                  + properties.length);
        }
        propertyResults.add(properties);
      }
      reducePropertyBooleans(propertyResults, signature);
    } else {
      signature.addAll(Collections.nCopies(NUM_BOOL_PROPERTIES, PropertySummary.TYPE_MISMATCH));
    }
  }

  private static void processComparison(
      Value value, Value output, List<PropertySummary> signature) {
    Class<?> type = value == null ? void.class : value.getType();
    int numExamples = output.getWrappedValues().size();

    if (type.equals(String.class)) {
      List<boolean[]> propertyResults = new ArrayList<>(); // num examples x num properties
      for (int i = 0; i < numExamples; i++) {
        String str = (String) value.getWrappedValue(i);
        String lower = Ascii.toLowerCase(str);
        String outputStr = (String) output.getWrappedValue(i);
        String outputStrLower = Ascii.toLowerCase(outputStr);

        boolean[] properties =
            new boolean[] {
              outputStr.contains(str), // output contains input?
              outputStr.startsWith(str), // output starts with input?
              outputStr.endsWith(str), // output ends with input?
              str.contains(outputStr), // input contains output?
              str.startsWith(outputStr), // input starts with output?
              str.endsWith(outputStr), // input ends with output?
              outputStrLower.contains(lower), // output contains input ignoring case?
              outputStrLower.startsWith(lower), // output starts with input ignoring case?
              outputStrLower.endsWith(lower), // output ends with input ignoring case?
              lower.contains(outputStrLower), // input contains output ignoring case?
              lower.startsWith(outputStrLower), // input starts with output ignoring case?
              lower.endsWith(outputStrLower), // input ends with output ignoring case?
              str.equals(outputStr), // input equals output?
              lower.equals(outputStrLower), // input equals output ignoring case?
              str.length() == outputStr.length(), // input same length as output?
              str.length() < outputStr.length(), // input shorter than output?
              str.length() > outputStr.length(), // input longer than output?
            };
        if (properties.length != NUM_STRING_COMPARISON_PROPERTIES) {
          throw new SynthesisError(
              "NUM_STRING_COMPARISON_PROPERTIES is "
                  + NUM_STRING_COMPARISON_PROPERTIES
                  + " but should be "
                  + properties.length);
        }
        propertyResults.add(properties);
      }
      reducePropertyBooleans(propertyResults, signature);
    } else {
      signature.addAll(
          Collections.nCopies(NUM_STRING_COMPARISON_PROPERTIES, PropertySummary.TYPE_MISMATCH));
    }

    if (type.equals(Integer.class)) {
      List<boolean[]> propertyResults = new ArrayList<>(); // num examples x num properties
      for (int i = 0; i < numExamples; i++) {
        int integer = (int) value.getWrappedValue(i);
        String outputStr = (String) output.getWrappedValue(i);
        boolean[] properties =
            new boolean[] {
              integer < outputStr.length(), // is less than output length?
              integer <= outputStr.length(), // is less or equal to output length?
              integer == outputStr.length(), // is equal to output length?
              integer >= outputStr.length(), // is greater or equal to output length?
              integer > outputStr.length(), // is greater than output length?
              Math.abs(integer - outputStr.length()) <= 1, // is very close to output length?
              Math.abs(integer - outputStr.length()) <= 3, // is close to output length?
            };
        if (properties.length != NUM_INT_COMPARISON_PROPERTIES) {
          throw new SynthesisError(
              "NUM_INT_COMPARISON_PROPERTIES is "
                  + NUM_INT_COMPARISON_PROPERTIES
                  + " but should be "
                  + properties.length);
        }
        propertyResults.add(properties);
      }
      reducePropertyBooleans(propertyResults, signature);
    } else {
      signature.addAll(
          Collections.nCopies(NUM_INT_COMPARISON_PROPERTIES, PropertySummary.TYPE_MISMATCH));
    }
  }

  private static void reducePropertyBooleans(
      List<boolean[]> propertyResults, List<PropertySummary> signature) {
    int numExamples = propertyResults.size();
    int numProperties = propertyResults.get(0).length;
    for (int i = 0; i < numProperties; i++) {
      boolean hasTrue = false;
      boolean hasFalse = false;
      for (int j = 0; j < numExamples; j++) {
        if (propertyResults.get(j)[i]) {
          hasTrue = true;
        } else {
          hasFalse = true;
        }
      }
      if (hasTrue && hasFalse) {
        signature.add(PropertySummary.MIXED);
      } else if (hasTrue && !hasFalse) {
        signature.add(PropertySummary.ALL_TRUE);
      } else if (!hasTrue && hasFalse) {
        signature.add(PropertySummary.ALL_FALSE);
      } else {
        throw new SynthesisError("A boolean[] was nonempty but contained no true or false?");
      }
    }
  }

  private ComputeSignature() {}
}
